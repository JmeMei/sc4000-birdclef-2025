# R1: 2nd-place pretrained backbones (Sydorskyi)

Repo: https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place (raw files read via raw.githubusercontent.com, main branch)
Dataset: https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-all-pretrained-models (page is JS-rendered; file list NOT retrieved; no shell/kaggle CLI was available during this lookup)

## (a) Contents
- README: "Pretrained backbones for fine-tuning". Filenames not described.
- Only path evidence: selected_ebs.py uses
  `data/birdclef_2025_all_pretrained_models/models_2024/tf_efficientnetv2_s_in21k_Pretrainversion1.pth`
- Backbones used anywhere in repo: `tf_efficientnetv2_s_in21k` ("ebs") and `eca_nfnet_l0` ("eca"). NO EfficientNet-B0 found in repo. Closest: tf_efficientnetv2_s_in21k (much bigger, ~21M params) or eca_nfnet_l0 (~24M). Whether the dataset holds anything else: UNCONFIRMED. Run `kaggle datasets files vladimirsydor/bird-clef-2025-all-pretrained-models` (check --help) to confirm.
- Pretraining: pretrain_ebs_2025.py head num_class = 7544 (Xeno-canto/iNat-style large label set); fine-tune configs use 206.
- Extraction: scripts/create_pretrain_backbone.py does `torch.load(in, map_location="cpu")["state_dict"]`, keeps keys starting `model.backbone.`, strips that prefix, `torch.save` of OrderedDict. So a backbone file should be a plain state dict with timm keys (e.g. conv_stem.weight, blocks.*, bn2.*), no head keys. README example output name: last_backbone.ckpt. Whether the dataset .pth files follow this: UNCONFIRMED (inspect with torch.load).
- Backbone is built with `timm.create_model(name, pretrained=..., in_chans=1, features_only=True, out_indices=..., exportable=...)`. features_only changes key naming slightly for some archs; check keys when loading (strict=False and print missing/unexpected).
- Not to use: dataset `bird-clef-2025-models` = finished trained models (OpenVINO). Also bird-clef-2025-pretrained-metadata, -add-data, -pseudo, -code-final, -addones are other datasets.

## (b) Front-end (pretrain_ebs_2025.py, selected_ebs.py)
- sample_rate 32000, n_fft 2048, hop_length 512, n_mels 128, f_min 20, normalized=True (torchaudio MelSpectrogram flag, i.e. frame-normalised STFT)
- f_max not set -> TraceableMelspec passes None to MelScale (torchaudio default = sr/2 = 16000; UNCONFIRMED exactly)
- power 2.0, Hann window, center=True, reflect pad, htk mel scale, norm=None
- dB: ChannelAgnosticAmplitudeToDB, stype power, 10*log10, amin 1e-10, top_db None (no clipping)
- Then NormalizeMelSpec: per-sample (X-mean)/(std+1e-6) over (freq,time), then per-sample min-max to [0,1]
- Input channels: in_chans = number of spectrograms = 1
- Train window 5 s; head AttHead (p 0.5, hidden 1024)
- Optional spec resize (spec_resize) exists; whether used: UNCONFIRMED
- Note: the repo takes waveform input and mel is inside the model; precompute_features.py only stores resampled 32 kHz mono waveforms in HDF5 (no mel).

## (c) Loading into our timm model
- Our v2: EfficientNet-B0, 128 mels, n_fft 2048, hop 512, 50-14000 Hz, dB rel. window max, 80 dB range, [0,1]. Matches pretraining in n_fft/hop/n_mels; differs in fmin (20 vs 50), fmax (16000 vs 14000), dB normalisation (per-sample z-score + min-max vs max-ref 80 dB). Sample rate of v2 cache: check our team's baseline (not looked up here).
- No B0 weights exist, so a drop-in swap into B0 is impossible; step 1 needs the backbone arch changed to tf_efficientnetv2_s_in21k (or eca_nfnet_l0): `m = timm.create_model("tf_efficientnetv2_s_in21k", pretrained=False, in_chans=1, num_classes=0)`; `sd = torch.load(path, map_location="cpu")`; if wrapped use `sd["state_dict"]` and strip `model.backbone.`; `m.load_state_dict(sd, strict=False)` and inspect missing/unexpected keys (features_only key naming may need unwrapping). Loading code for backbone_path is not in the fetched files (lives in lightning_training / other code_base module); UNCONFIRMED.
- in_chans=1 vs our baseline (ImageNet 3-ch or 1-ch): check; the pretrained conv_stem has 1 input channel.

## Sources
- https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place (README, train_configs/pretrain_ebs_2025.py, selected_ebs.py, scripts/create_pretrain_backbone.py, scripts/precompute_features.py, code_base/models/wave_clasifier.py, blocks.py)
- https://arxiv.org/html/2507.08236v1 / CEUR Vol-4283 paper330: a later paper says it loaded these Xeno-Canto-pretrained 2025 checkpoints (EffNetV2-S, eca_nfnet_l0) via timm (secondary, from search snippet).
