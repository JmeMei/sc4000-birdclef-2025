# R1b: conv_head / bn2 in 2nd-place tf_efficientnetv2_s_in21k

Sources (raw GitHub):
- https://raw.githubusercontent.com/VSydorskyy/BirdCLEF_2025_2nd_place/main/code_base/models/wave_clasifier.py
- .../code_base/models/blocks.py
- .../train_configs/pretrain_ebs_2025.py
- https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/timm/models/efficientnet.py

(Fetched via a summarizing tool; quotes are as returned, not read line-by-line.)

## 1. Features only, no conv_head/bn2
wave_clasifier.py:
    self.backbone = timm.create_model(backbone, features_only=True, pretrained=pretrained,
        exportable=exportable, in_chans=self._n_specs, out_indices=deep_supervision_steps, **add_backbone_config_)
    deep_supervision_steps: Optional[List[int]] = None   (default)
    backbone_channels = self.backbone.feature_info.channels()
    ... else: emb = self.backbone(spec)[-1]
    in_chans=(backbone_channels[-1] if deep_supervision_steps is None else backbone_channels)
Config pretrain_ebs_2025.py: backbone "tf_efficientnetv2_s_in21k", no deep_supervision_steps set -> None -> timm default out_indices -> last stage map. Last channel = 256 for tf_efficientnetv2_s (UNVERIFIED here; from timm knowledge, stride 32, channels 24/48/64/160/256).
Config also: sample_rate 32000, n_mels 128, f_min 20, n_fft 2048, hop 512, head hidden_chans 1024, p 0.5, num_class 7544.

## 2. Head (blocks.py AttHead, SED-style)
- GeM pooling over frequency: `self.pooling = GeMFreq(...)`
- `nn.Dropout(p / 2)`, `nn.Linear(in_chans, hidden_chans)`, `nn.ReLU()`, `nn.Dropout(p)`
- `self.attention = nn.Conv1d(hidden_chans, num_class, kernel_size=1)` and `self.fix_scale = nn.Conv1d(..., num_class, kernel_size=1, bias=True)`
- `time_att = torch.tanh(self.attention(feat))`; `feat_v = self.fix_scale(feat)`;
  clipwise = sum(sigmoid(feat_v) * softmax(time_att, dim=-1)) over time.
Exact class used (AttHead vs AttHeadSimplified) not confirmed from the config fetch.

## 3. timm
EfficientNet.__init__ (num_features > 0):
    self.conv_head = create_conv2d(head_chs, num_features, 1, padding=pad_type, **dd)
    self.bn2 = norm_act_layer(num_features, inplace=True, **dd)
forward_features: `x = self.conv_head(x)` then `x = self.bn2(x)` (so in forward_features, not forward_head).
forward_head: global_pool then classifier only.
EfficientNetFeatures: defines conv_stem, bn1, blocks, feature_info; never defines/calls conv_head or bn2.
1x1 conv to num_features=1280 for v2_s: 1280 not seen in fetched text (UNVERIFIED; standard timm default num_features=1280).
Model name in timm is tf_efficientnetv2_s.in21k (legacy alias tf_efficientnetv2_s_in21k).

## 4. Conclusion
By construction (features_only=True), their backbone state dict should contain only conv_stem, bn1, blocks.* keys, no conv_head.* / bn2.*. NOT verified by inspecting the actual checkpoint; check by loading the Kaggle dataset weights and listing keys.
Implication: loading into a full timm EfficientNet (the B0 equivalent in our team's baseline) needs strict=False for head keys; conv_head/bn2 stay ImageNet-init or must be dropped.
