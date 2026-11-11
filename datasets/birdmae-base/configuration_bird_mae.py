from transformers import PretrainedConfig
from typing import Literal


class BirdMAEConfig(PretrainedConfig):
    """This represents the Bird-MAE-Base config from the original paper"""
    _auto_class = "AutoConfig"

    def __init__(
            self,
            img_size_x: int = 512,
            img_size_y: int = 128,
            patch_size: int = 16,
            in_chans: int = 1,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: int = 4,
            pos_trainable: bool = False,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: float = None,
            drop_rate: float = 0.0,
            norm_layer_eps: float = 1e-6,
            global_pool: Literal["cls", "mean"] | None = "mean",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.img_size_x = img_size_x
        self.img_size_y = img_size_y
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.pos_trainable = pos_trainable

        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.init_values = init_values
        self.drop_rate = drop_rate
        self.pos_drop_rate = drop_rate
        self.attn_drop_rate = drop_rate
        self.drop_path_rate = drop_rate
        self.proj_drop_rate = drop_rate
        self.norm_layer_eps = norm_layer_eps
        self.global_pool = global_pool

        # Calculated properties (useful for initializing the model)
        self.num_patches_x = img_size_x // patch_size
        self.num_patches_y = img_size_y // patch_size
        self.num_patches = self.num_patches_x * self.num_patches_y
        self.num_tokens = self.num_patches + 1