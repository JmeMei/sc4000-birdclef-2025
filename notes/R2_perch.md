# R2 Perch 2.0 findings

Official: Kaggle Models google/bird-vocalization-classifier, TF2 variant perch_v2 (and perch_v2_cpu per third-party mention; exact version number UNVERIFIED, Kaggle page not fetchable).
 https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2
Paper: arXiv 2508.04665. Licence: Apache-2.0 (per HF cards; Kaggle page licence not verified).

Input (ONNX, community, https://huggingface.co/Calotriton/perch-v2-onnx): `inputs` float32 [batch,160000] = 5 s mono 32 kHz; preprocessing: polyphase resample, per-window remove mean, scale peak to 0.25.
Outputs: label [B,14795] logits; embedding [B,1536]; spatial_embedding [B,16,4,1536]; spectrogram [B,500,128]. File perch_v2.onnx 409 MB, labels.csv, perch_v2_ebird_classes.csv. ORT CPU matched TF within ~7e-6.
Alt ONNX/TFLite: https://huggingface.co/justinchuby/Perch-onnx (perch_v2.onnx, perch_v2_no_dft.onnx faster/slightly less accurate, perch_v2.tflite). README benchmark: ONNX ~66 ms/run, TFLite ~609 ms/run (hardware and batch size not stated -> unverified).
Alt: https://huggingface.co/tphakala/Perch-v2-Models (fp32 409MB, no_dft 413MB, int8_arm 131MB, regional slices). Also PyTorch port https://huggingface.co/matijama/perch-v2-pytorch (unchecked).
Throughput estimate: 700*12=8400 windows * ~66 ms ~ 9 min if 66 ms is per 5 s window on CPU (unverified).
TF version: Calotriton reference outputs from perch_v2_cpu on TF 2.21 (third-party). Kaggle image TF version not verified.
Note: a 2026 paper mentions 1280-d / 2333 classes, which is a different head/version; v2 here is 1536-d.
Other: BirdCLEF 2026 papers https://ceur-ws.org/Vol-4283/paper321.pdf , paper359.pdf (mention TFLite FP16 with SELECT_TF_OPS on CPU).
Unverified: exact Kaggle signature/output keys (TF SavedModel; expect "embedding","label" but not confirmed), Kaggle load snippet.
