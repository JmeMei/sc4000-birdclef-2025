import numpy as np
import collections
from itertools import repeat
from functools import partial
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from .configuration_bird_mae import BirdMAEConfig

logger = logging.get_logger(__name__)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed_flexible(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32) # grid size[0] = 8
    grid_w = np.arange(grid_size[1], dtype=np.float32) # grid size[1] = 32
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0) # 2,8,32

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]]) # 2,1,8.32
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed # 267 (+cls) x 1024 (feature dim)

# From timm.models.layers
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _ntuple(2)(bias)
        drop_probs = _ntuple(2)(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# Modified from timm.models.vision_transformer
class Attention(nn.Module):
    """Standard Multi-head Self Attention module with QKV projection.

    This module implements the standard multi-head attention mechanism used in transformers.
    It supports both the fused attention implementation (scaled_dot_product_attention) for
    efficiency when available, and a manual implementation otherwise. The module includes
    options for QK normalization, attention dropout, and projection dropout.
    """
    fused_attn: bool = True

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = None,
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: torch.Tensor = None,
            output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn_weights = None

        if self.fused_attn and not output_attentions:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn if attn_mask is None else attn + attn_mask
            attn_weights = attn.softmax(dim=-1)
            x = self.attn_drop(attn_weights) @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_weights


# From timm.models.vision_transformer
class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: float = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor,
                output_attentions: bool = False,
                attn_mask: torch.Tensor = None
                ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        #x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x_skip = x
        x = self.norm1(x)
        x, att = self.attn(x, output_attentions=output_attentions, attn_mask=attn_mask)
        x = self.ls1(x)
        x = self.drop_path1(x)
        x += x_skip
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, att

# From timm.models.vision_transformer
class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchEmbed_org(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self,
                 img_size: int | tuple[int, ...] = 224,
                 patch_size: int | tuple[int, ...] = 16,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size: tuple[int,int] = _ntuple(2)(img_size) # audio mae used: (target_length x 128) --> not sure why tbh
        patch_size: tuple[int,int] = _ntuple(2)(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0]) # number of patches height/width = 8/32
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape #batch size, channels, height, width --> apparently sth else is expected???
        x = self.proj(x) # 1, 1, 512, 128 -> 1, 768, 32, 8 (batch, 768 channel, 32 height, 8 width)
        x = x.flatten(2) # 1, 768, 32, 8 -> 1, 768, 256
        x = x.transpose(1, 2) # 1, 768, 256 -> 1, 256, 768
        return x

# --- END OF NECESSARY TIMM/Custom internal module definitions ---


class BirdMAEPreTrainedModel(PreTrainedModel):
    config_class = BirdMAEConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            w = module.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


class BirdMAEModel(BirdMAEPreTrainedModel):
    _auto_class = "AutoModel"
    #_keys_to_ignore_on_load_missing = ["fc_norm.weight", "fc_norm.bias"]

    def __init__(self, config: BirdMAEConfig):
        super().__init__(config)

        self.patch_embed = PatchEmbed_org(
            img_size=(config.img_size_x, config.img_size_y),  # (512, 128)
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.embed_dim),
            requires_grad=config.pos_trainable
        )

        if self.pos_embed.data.shape[1] == config.num_patches + 1:
            pos_embed_np = get_2d_sincos_pos_embed_flexible(
                self.pos_embed.shape[-1],  # embedding dim
                self.patch_embed.patch_hw,  # (8, 32) for a 128x512 image with 16x16 patches
                cls_token=True
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed_np).float().unsqueeze(0))
        else:
            logger.warning("Positional embedding shape mismatch. Will not initialize sin-cos pos embed.")

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                qk_norm=config.qk_norm,
                init_values=config.init_values,
                proj_drop=config.proj_drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[i],
                #norm_layer=nn.LayerNorm(config.embed_dim, eps=config.norm_layer_eps)
                norm_layer=partial(nn.LayerNorm, eps=config.norm_layer_eps)
            )
            for i in range(config.depth)
        ])

        self.pos_drop = nn.Dropout(p=config.pos_drop_rate)
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.norm_layer_eps) #norm_layer(config.embed_dim)
        self.fc_norm = nn.LayerNorm(config.embed_dim, eps=config.norm_layer_eps) #norm_layer(config.embed_dim)
        self.global_pool = config.global_pool

        nn.init.trunc_normal_(self.cls_token, std=.02)

    def forward(
            self,
            input_values : torch.Tensor,
            attention_mask: torch.Tensor = None,
            output_attentions: bool = None,
            output_hidden_states: bool = None,
            return_dict: bool = None,
    ) -> tuple | BaseModelOutput:
        if len(input_values.shape) == 3:
            input_values = input_values.unsqueeze(0)

        output_attentions = output_attentions or self.config.output_attentions

        output_hidden_states = output_hidden_states or self.config.output_hidden_states
        return_dict = return_dict or self.config.use_return_dict

        B, C, X, Y = input_values.shape
        assert X == self.config.img_size_x, f"Expected image_size_x={self.config.img_size_x} but was {X}."
        assert Y == self.config.img_size_y, f"Expected image_size_y={self.config.img_size_y} but was {Y}."

        x = self.patch_embed(input_values)

        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        all_hidden_states = (x,) if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for blk in self.blocks:
            x, self_attn_weights = blk(x, output_attentions=output_attentions, attn_mask=attention_mask)
            if output_hidden_states:
                all_hidden_states += (x,)
            if output_attentions:
                all_self_attns += (self_attn_weights,)

        if self.global_pool is None:
            pooled_output = x
        elif self.global_pool == "mean":
            x = x[:, 1:, :].mean(dim=1)
            pooled_output = self.fc_norm(x)
        elif self.global_pool == "cls":
            x = self.norm(x)
            pooled_output = x[:, 0]
        else:
            raise ValueError(f"Invalid global pool type: {self.global_pool}")

        if not return_dict:
            return (pooled_output,) + (all_hidden_states if output_hidden_states else ()) + (None,)

        return BaseModelOutput(
            last_hidden_state=pooled_output,
            hidden_states=all_hidden_states,
            attentions=all_self_attns
        )


class BirdMAEForSequenceClassification(BirdMAEPreTrainedModel):
    _auto_class = "AutoModelForSequenceClassification"
    def __init__(self, config: BirdMAEConfig, head_type: Literal["linear", "ppnet"]):
        super().__init__(config)
        self.num_labels = self.config.num_labels
        self.head_type = head_type
        self.model = BirdMAEModel(config)
        if head_type == "linear":
            self.head = nn.Linear(config.embed_dim, self.num_labels, bias=False)
        elif head_type == "ppnet":
            pass
        else:
            raise NotImplementedError(f"{head_type=} is not supported.")

    def forward(self,
                input_values: torch.Tensor,
                attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None,
                output_attentions: bool = None,
                output_hidden_states: bool = None,
                return_dict: bool = None):
        return_dict = return_dict or self.config.return_dict
        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = output_hidden_states or self.config.output_hidden_states

        output = self.model(input_values,
                            attention_mask=attention_mask,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)

        hidden_state = output[0]
        logits = self.head(hidden_state)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    raise NotImplementedError(f"Setting num_labels={self.num_labels} indicates a regression task, which is not supported.")
                elif self.num_labels > 1 and labels.shape != logits.shape:
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )






