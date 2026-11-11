# R3 Bird-MAE-Base

Sources: https://huggingface.co/DBD-research-group/Bird-MAE-Base (README, config.json, preprocessor_config.json, feature_extractor.py, modeling_bird_mae.py via /raw/main/), https://arxiv.org/abs/2504.12880, https://arxiv.org/html/2504.12880 (first 100k chars read), https://github.com/DBD-research-group/Bird-MAE

## Loading
    from transformers import AutoFeatureExtractor, AutoModel
    model = AutoModel.from_pretrained("DBD-research-group/Bird-MAE-Base", trust_remote_code=True)
    fe = AutoFeatureExtractor.from_pretrained("DBD-research-group/Bird-MAE-Base", trust_remote_code=True)
    audio, sr = librosa.load(path, sr=32000); x = fe(audio); emb = model(x)
- config.json says transformers_version 4.38.0 (min version not stated). No timm needed. Python >= 3.10 (type hints).
- Remote code files: configuration_bird_mae.py, feature_extractor.py, modeling_bird_mae.py.

## Input
- 32 kHz, 5 s (160000 samples) clip, padded/truncated. Kaldi fbank (torchaudio): 128 mel bins, Hann, htk_compat, 10 ms shift, dither 0; n_fft/frame length not explicit in code (paper says ~800). 512 frames; shorter padded with min value.
- Normalise (x - (-7.2)) / (4.43*2). Output tensor [B, 1, 512, 128]. in_chans=1, patch 16, 256 patches (32x8), ViT-B 12 layers, 768-d, 85.5M params.
- BirdCLEF 5 s windows match the native length exactly; cut raw audio at 32 kHz (resample from 32 kHz OGG, no resampling needed). Cannot reuse the baseline mel cache.

## Output / probes
- config global_pool = "mean": forward returns (B, 768) = mean of patch tokens (no CLS) + fc_norm. Options: None (full tokens, CLS at idx 0, 257 tokens), "cls".
- Paper: linear probe = globally average-pooled patch features (not CLS). Prototypical probe = full patch map, cosine similarity to J prototypes/class (J=20 main, tested 5-30), max over patches. Proto beats linear by up to 37 pts MAP.
- Prototypical probing head exists in the repo (README says Linear and Proto head options) - exact class not verified.

## Offline packaging
- Files (342 MB total): model.safetensors (342 MB), config.json, preprocessor_config.json, configuration_bird_mae.py, feature_extractor.py, modeling_bird_mae.py (+README). Snapshot the folder to a Kaggle dataset, load with from_pretrained(local_path, trust_remote_code=True). Not tested offline.

## Licence
- UNVERIFIED: no license in HF model card header, none in GitHub sidebar/file listing. Check repo LICENSE / HF metadata manually.

## CPU cost
- Not published. ViT-B on 256 tokens: roughly 17.5 GFLOPs/forward (our estimate from standard ViT-B at 197 tokens ~17.6 GFLOPs, scaled ~1.3x = ~23 GFLOPs); UNVERIFIED, must profile. 700 files x 12 windows = 8400 windows.
- Page disclaimer: "There might be some error in the models, we have to check it."
