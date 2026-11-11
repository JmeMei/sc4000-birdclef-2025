# Kaggle notebook catalogue

This file lists the 40 Kaggle notebooks we created under the Kaggle account `strippedbanana` for the BirdCLEF+ 2025 course project (NTU SC4000, late submissions), with a paste-ready description for each. All 40 are private, and their kernel slugs start with `sc4000-birdclef-m3-`. On 10 Nov 2026 we ran `kaggle kernels status` for all 40 and every one reported COMPLETE. The account's kernel list contains exactly these 40, and we read the 19 competition submissions and 2 private datasets from the Kaggle CLI the same day. Figures come from `reports/results_table.md`, `reports/timing_table.md`, `STATE.md`, `docs/LOG.md` and the run outputs in `docs/_kaggle_logs/`. Where a value was not logged it says "not recorded". CPU minutes are for 700 one-minute soundscapes; "projected" means scaled up from a 20-file dry run.

## Quick index

| Phase | Notebook | Accelerator | Submission ref |
|---|---|---|---|
| Baseline and pretrained models (incl. speed) | [sc4000-birdclef-m3-a1-v2s-train](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-v2s-train) | GPU (NVIDIA T4) | none |
| Baseline and pretrained models (incl. speed) | [sc4000-birdclef-m3-a1-v2s-infer](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-v2s-infer) | CPU only | 58120958 |
| Baseline and pretrained models (incl. speed) | [sc4000-birdclef-m3-a2-perch-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-embed) | GPU (NVIDIA T4) | none |
| Baseline and pretrained models (incl. speed) | [sc4000-birdclef-m3-a2-perch-infer](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-infer) | CPU only | 58122219 |
| Baseline and pretrained models (incl. speed) | [sc4000-birdclef-m3-a3-birdmae-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed) | GPU (NVIDIA T4) | none |
| Baseline and pretrained models (incl. speed) | [sc4000-birdclef-m3-a3-probe-retry](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-probe-retry) | CPU only | none |
| Baseline and pretrained models (incl. speed) | [sc4000-birdclef-m3-a3-mlp-head](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-mlp-head) | CPU only | none |
| Baseline and pretrained models (incl. speed) | [sc4000-birdclef-m3-a3-birdmae-infer](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-birdmae-infer) | CPU only | 58127656 |
| Baseline and pretrained models (incl. speed) | [sc4000-birdclef-m3-profile-infer-v2](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-profile-infer-v2) | CPU only | none |
| Baseline and pretrained models (incl. speed) | [sc4000-birdclef-m3-fast-infer](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-fast-infer) | CPU only | 58159079 |
| Adaptation (AdaBN, CORAL, bug check) | [sc4000-birdclef-m3-a1-adabn](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-adabn) | GPU (NVIDIA T4) | none |
| Adaptation (AdaBN, CORAL, bug check) | [sc4000-birdclef-m3-a1-adabn-fast-infer](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-adabn-fast-infer) | CPU only | 58158788 |
| Adaptation (AdaBN, CORAL, bug check) | [sc4000-birdclef-m3-v2-adabn](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-v2-adabn) | GPU (NVIDIA T4) | none |
| Adaptation (AdaBN, CORAL, bug check) | [sc4000-birdclef-m3-v2-adabn-fast-infer](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-v2-adabn-fast-infer) | CPU only | 58162933 |
| Adaptation (AdaBN, CORAL, bug check) | [sc4000-birdclef-m3-soundscape-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-soundscape-embed) | GPU (NVIDIA T4) | none |
| Adaptation (AdaBN, CORAL, bug check) | [sc4000-birdclef-m3-coral-heads](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-coral-heads) | CPU only | none |
| Adaptation (AdaBN, CORAL, bug check) | [sc4000-birdclef-m3-a2-coral-std-infer](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-coral-std-infer) | CPU only | 58160573 |
| Adaptation (AdaBN, CORAL, bug check) | [sc4000-birdclef-m3-a2-meanstd-infer](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-meanstd-infer) | CPU only | 58160577 |
| Adaptation (AdaBN, CORAL, bug check) | [sc4000-birdclef-m3-a3-coral-std-infer](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-coral-std-infer) | CPU only | 58160580 |
| Adaptation (AdaBN, CORAL, bug check) | [sc4000-birdclef-m3-a3-meanstd-infer](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-meanstd-infer) | CPU only | 58160584 |
| Adaptation (AdaBN, CORAL, bug check) | [sc4000-birdclef-m3-coral-diag](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-coral-diag) | CPU only | none |
| Adaptation (AdaBN, CORAL, bug check) | [sc4000-birdclef-m3-head-seed-repeats](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-head-seed-repeats) | CPU only | none |
| Distillation | [sc4000-birdclef-m3-a3-distill](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-distill) | CPU only | none |
| Distillation | [sc4000-birdclef-m3-a3-distill-infer](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-distill-infer) | CPU only | 58161782 |
| Distillation | [sc4000-birdclef-m3-a3-distill-coral-infer](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-distill-coral-infer) | CPU only | 58162938 |
| Distillation | [sc4000-birdclef-m3-a2-a1distill](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-a1distill) | GPU (NVIDIA T4) | none |
| Distillation | [sc4000-birdclef-m3-a2-a1distill-infer](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-a1distill-infer) | CPU only | 58164973 |
| Noisy student | [sc4000-birdclef-m3-ns-perch-all](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-perch-all) | GPU (NVIDIA T4) | none |
| Noisy student | [sc4000-birdclef-m3-ns-a1-all](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-a1-all) | GPU (NVIDIA T4) | none |
| Noisy student | [sc4000-birdclef-m3-ns-pseudo-labels](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-pseudo-labels) | CPU only | none |
| Noisy student | [sc4000-birdclef-m3-ns-mel-check](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-mel-check) | GPU (NVIDIA T4) | none |
| Noisy student | [sc4000-birdclef-m3-ns-student-main](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-main) | GPU (NVIDIA T4) | none |
| Noisy student | [sc4000-birdclef-m3-ns-student-ablation](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-ablation) | GPU (NVIDIA T4) | none |
| Noisy student | [sc4000-birdclef-m3-ns-student-main-infer](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-main-infer) | CPU only | 58181262 |
| Noisy student | [sc4000-birdclef-m3-ns-student-ablation-infer](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-ablation-infer) | CPU only | 58181267 |
| Noisy student | [sc4000-birdclef-m3-ns-head-seeds](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-head-seeds) | CPU only | none |
| Ensemble | [sc4000-birdclef-m3-ens-e1](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ens-e1) | CPU only | 58196484 |
| Ensemble | [sc4000-birdclef-m3-ens-e2](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ens-e2) | CPU only | 58196486 |
| Ensemble | [sc4000-birdclef-m3-ens-e3](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ens-e3) | CPU only | 58198886 |
| Ensemble | [sc4000-birdclef-m3-ens-e4](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ens-e4) | CPU only | 58198889 |

## 1. Baseline and pretrained models (incl. speed)

### sc4000-birdclef-m3-a1-v2s-train

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-v2s-train
- Local folder: `notebooks/a1_v2s_train/`
- Accelerator: GPU (NVIDIA T4). Internet on.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: none
- Inputs: dataset [vladimirsydor/bird-clef-2025-all-pretrained-models](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-all-pretrained-models); kernel output [jmemei/sc4000-birdclef-m1-make-mels](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-make-mels)
- Key outputs/result: Best validation ROC-AUC 0.9832 (epoch 9, measured); about 0.5 T4 GPU hours (0.45 h of epochs plus setup, as logged in STATE.md).
- Kaggle description (paste-ready):

  > Fine-tunes an EfficientNetV2-S image network on log-mel spectrograms of the 206 BirdCLEF+ 2025 species, starting from weights that the 2nd-place BirdCLEF 2025 team pretrained on bird audio. It reuses our team's baseline mel cache and the same recipe as the course baseline (random 5 s crops, mixup, 15 epochs, stratified 80/20 split with seed 42). Best ROC-AUC (area under the ROC curve) on the held-out 20% is 0.9832. The saved weights are used by the A1 inference notebooks, which scored 0.82822 public / 0.81642 private.

### sc4000-birdclef-m3-a1-v2s-infer

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-v2s-infer
- Local folder: `notebooks/a1_v2s_infer/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: 58120958, public 0.82822 / private 0.81642
- Inputs: kernel output [strippedbanana/sc4000-birdclef-m3-a1-v2s-train](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-v2s-train)
- Key outputs/result: Dry run on 20 files projects to about 37 CPU min per 700 files (projected, likely pessimistic: it includes pool start-up spread over only 20 files). Scored without a timeout.
- Kaggle description (paste-ready):

  > Predicts every 5 s window of the hidden BirdCLEF+ 2025 soundscapes on CPU using the EfficientNetV2-S weights trained in the A1 training notebook. Audio is turned into log-mel spectrograms (frequency-scaled spectrograms) and normalised per window before the network runs. The submission scored 0.82822 public / 0.81642 private, against 0.763 / 0.789 for the course baseline. A 20-file dry run projects to about 37 minutes for 700 files.

### sc4000-birdclef-m3-a2-perch-embed

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-embed
- Local folder: `notebooks/a2_perch_embed/`
- Accelerator: GPU (NVIDIA T4). Internet on.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: none
- Inputs: kernel output [jmemei/sc4000-birdclef-m1-make-mels](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-make-mels)
- Key outputs/result: 106,157 windows from 28,564 train_audio clips embedded in 3024 s wall (35.1 windows/s, measured); MLP head val AUC 0.9916 versus 0.9071 for logistic regression; about 0.9 to 1.0 T4 GPU hours.
- Kaggle description (paste-ready):

  > Runs Perch 2.0, a bird-sound model from Google, frozen on the 5 s windows (up to 30 s per clip) of the 28,564 training clips and keeps its 1536-number embedding for each window. It then trains a small neural-network head on those embeddings with the same split as the baseline; the head reaches 0.9916 ROC-AUC on the held-out 20%, against 0.9071 for logistic regression. Embeddings are cached so they are computed once. Perch 2.0 was released after BirdCLEF 2025 closed.

### sc4000-birdclef-m3-a2-perch-infer

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-infer
- Local folder: `notebooks/a2_perch_infer/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: 58122219, public 0.84774 / private 0.85226
- Inputs: dataset [strippedbanana/sc4000-birdclef-m3-perch-onnx](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-perch-onnx); kernel output [strippedbanana/sc4000-birdclef-m3-a2-perch-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-embed)
- Key outputs/result: 25.0 CPU min per 700 files (projected from a 20-file dry run, ONNX Runtime). Scored without a timeout.
- Kaggle description (paste-ready):

  > CPU inference for the A2 submission: it runs a community ONNX conversion of Perch 2.0 on each 5 s window of the hidden soundscapes and applies the small head trained in the embedding notebook. The submission scored 0.84774 public / 0.85226 private. A 20-file dry run projects to about 25 minutes for 700 files. Perch 2.0 was released after BirdCLEF 2025 closed.

### sc4000-birdclef-m3-a3-birdmae-embed

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed
- Local folder: `notebooks/a3_birdmae_embed/`
- Accelerator: GPU (NVIDIA T4). Internet on.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: none
- Inputs: kernel output [jmemei/sc4000-birdclef-m1-make-mels](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-make-mels)
- Key outputs/result: 106,157 windows from 28,564 clips embedded in 691 s wall (measured); linear probe first-5s val AUC 0.7495 (not used for any submission; see a3-probe-retry for why it was low); about 0.3 T4 GPU hours (estimated, not measured exactly).
- Kaggle description (paste-ready):

  > Extracts frozen embeddings from Bird-MAE-Base, a bird-audio masked-autoencoder model, for the same 5 s windows of the 28,564 training clips used in the Perch notebook, and caches them to disk. It also fits a first linear probe on them, which scores only 0.7495 first-5s validation ROC-AUC because of a bias-initialisation problem diagnosed in the following notebooks. The embeddings feed the Bird-MAE heads and the A3 submission.

### sc4000-birdclef-m3-a3-probe-retry

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-probe-retry
- Local folder: `notebooks/a3_probe_retry/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026). Version 1 = head sweep, version 2 = bias-initialisation test.
- Submission: none
- Inputs: kernel output [strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed); kernel output [jmemei/sc4000-birdclef-m1-make-mels](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-make-mels)
- Key outputs/result: First-5s val AUC: linear LR 1e-3 log-prior bias 0.9626, LR 3e-3 log-prior bias 0.9618, linear default bias 0.7498 (validation only, not submitted).
- Kaggle description (paste-ready):

  > Re-tests the linear probe on cached Bird-MAE embeddings to find out why it scored so low. Starting each class bias at the log of its frequency (about -6) lifts the linear probe from 0.7498 to 0.9626 first-5s validation ROC-AUC, so the earlier result came from bias initialisation under a low learning rate, not from the features. Nothing from this notebook was submitted.

### sc4000-birdclef-m3-a3-mlp-head

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-mlp-head
- Local folder: `notebooks/a3_mlp_head/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: none
- Inputs: kernel output [strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed); kernel output [jmemei/sc4000-birdclef-m1-make-mels](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-make-mels)
- Key outputs/result: MLP head first-5s val AUC 0.9749 (best epoch 26, saved); log-prior bias 0.9707. Run time 270 s (4.5 min, CPU).
- Kaggle description (paste-ready):

  > Trains the final head for Bird-MAE-Base: a small MLP (one hidden layer of 512 units, dropout 0.2) on the cached embeddings, with the same 80/20 split. The default bias start gives 0.9749 first-5s validation ROC-AUC and a log-prior bias start gives 0.9707, so the default is saved. It takes about 4.5 minutes on CPU, and its weights are used by the A3 inference notebook.

### sc4000-birdclef-m3-a3-birdmae-infer

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-birdmae-infer
- Local folder: `notebooks/a3_birdmae_infer/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: 58127656, public 0.77640 / private 0.78016
- Inputs: dataset [strippedbanana/sc4000-birdclef-m3-birdmae-base](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-birdmae-base); kernel output [strippedbanana/sc4000-birdclef-m3-a3-mlp-head](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-mlp-head); kernel output [strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed)
- Key outputs/result: 61.6 CPU min per 700 files (projected from a 20-file dry run, torch). Scored without a timeout.
- Kaggle description (paste-ready):

  > CPU inference for the A3 submission: it runs Bird-MAE-Base on each 5 s window of the hidden soundscapes and applies the MLP head trained on the training clips. The submission scored 0.77640 public / 0.78016 private, barely above the 0.776 public grading cut-off and below the baseline on private. A 20-file dry run projects to about 62 minutes for 700 files.

### sc4000-birdclef-m3-profile-infer-v2

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-profile-infer-v2
- Local folder: `notebooks/b0_profile_infer_v2/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: none
- Inputs: kernel output [jmemei/sc4000-birdclef-m1-train-v2](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-train-v2)
- Key outputs/result: Measured on 700 train_soundscapes files: 11.1 CPU min whole notebook (10.4 pipeline), 0.89 s per file; per file decode 0.320 s, mel 1.540 s, model 0.412 s.
- Kaggle description (paste-ready):

  > Times the course baseline's CPU inference (EfficientNet-B0) on 700 training soundscapes, split into audio decoding, mel-spectrogram computation and the network. The whole notebook takes 11.1 minutes; the mel step is the largest part at 1.54 s per file, against 0.32 s for decoding and 0.41 s for the model. The figure of about 25 minutes reported for our team's baseline was not reproduced here.

### sc4000-birdclef-m3-fast-infer

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-fast-infer
- Local folder: `notebooks/b_fast_infer/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026). Version 1 = v2 B0 timing run, version 2 = A1 V2-S timing run. The submitted control (ref 58159079) used this kernel; its version is not recorded.
- Submission: 58159079 (kernel version not recorded), public 0.82822 / private 0.81642
- Inputs: kernel output [jmemei/sc4000-birdclef-m1-train-v2](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-train-v2); kernel output [strippedbanana/sc4000-birdclef-m3-a1-v2s-train](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-v2s-train)
- Key outputs/result: Measured on 700 train_soundscapes files: v2 B0 5.5 CPU min whole notebook (4.0 pipeline); A1 V2-S 8.0 min (7.2 pipeline). Submitted control scored 0.82822 / 0.81642.
- Kaggle description (paste-ready):

  > A faster CPU inference pipeline: four worker processes with one thread each, one mel computation per file, and the 12 windows of a file run through the network as one batch. On 700 training soundscapes it cuts the baseline EfficientNet-B0 from 11.1 to 5.5 minutes, and runs the EfficientNetV2-S model (A1) in 8.0 minutes. Scoring the unadapted A1 model through it gave 0.82822 public / 0.81642 private, identical to the original A1 notebook, so the faster path does not change predictions.


## 2. Adaptation (AdaBN, CORAL, bug check)

### sc4000-birdclef-m3-a1-adabn

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-adabn
- Local folder: `notebooks/a1_adabn/`
- Accelerator: GPU (NVIDIA T4). Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026). Version 1 ended ERROR at its own CPU guard (projected 55.9 min against a 30 min limit); version 2 ran on a T4.
- Submission: none
- Inputs: kernel output [strippedbanana/sc4000-birdclef-m3-a1-v2s-train](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-v2s-train); kernel output [jmemei/sc4000-birdclef-m1-make-mels](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-make-mels)
- Key outputs/result: Val AUC 0.9832 before, 0.9713 after (12,000 windows, 187 batches, 110 BN modules); version 2 took 5.8 min, about 0.1 T4 GPU hours (measured).
- Kaggle description (paste-ready):

  > Applies AdaBN (re-estimating the batch-norm statistics on unlabelled soundscapes) to the A1 EfficientNetV2-S model, using 12,000 windows from 1000 training soundscapes. The idea is to reduce the gap between clean training clips and noisy field recordings without using labels. The held-out ROC-AUC on clean clips drops from 0.9832 to 0.9713, and the adapted model scored worse on the leaderboard (see the inference notebook).

### sc4000-birdclef-m3-a1-adabn-fast-infer

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-adabn-fast-infer
- Local folder: `notebooks/a1_adabn_infer/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: 58158788, public 0.77210 / private 0.76573
- Inputs: kernel output [strippedbanana/sc4000-birdclef-m3-a1-adabn](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-adabn)
- Key outputs/result: 10.7 CPU min per 700 files (measured, dry run on 700 train_soundscapes).
- Kaggle description (paste-ready):

  > Runs the AdaBN-adapted A1 model on the hidden soundscapes with the fast CPU pipeline. It scored 0.77210 public / 0.76573 private, about 0.056 and 0.051 below the unadapted A1 model (0.82822 / 0.81642), so re-estimating the batch-norm statistics on soundscapes hurt. The run takes 10.7 minutes for 700 files.

### sc4000-birdclef-m3-v2-adabn

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-v2-adabn
- Local folder: `notebooks/v2_adabn/`
- Accelerator: GPU (NVIDIA T4). Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: none
- Inputs: kernel output [jmemei/sc4000-birdclef-m1-train-v2](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-train-v2); kernel output [jmemei/sc4000-birdclef-m1-make-mels](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-make-mels)
- Key outputs/result: Val AUC 0.9766 before, 0.9405 after (49 BN modules); 5.1 min, about 0.09 T4 GPU hours (measured).
- Kaggle description (paste-ready):

  > Applies AdaBN (re-estimating the batch-norm statistics on unlabelled soundscapes) to the course baseline EfficientNet-B0 model, using 12,000 windows from training soundscapes. Held-out ROC-AUC on clean clips falls from 0.9766 to 0.9405. The purpose was to compare the effect of AdaBN on an ImageNet-pretrained backbone (this one) with its effect on the bird-pretrained A1 backbone.

### sc4000-birdclef-m3-v2-adabn-fast-infer

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-v2-adabn-fast-infer
- Local folder: `notebooks/v2_adabn_infer/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: 58162933, public 0.73889 / private 0.76383
- Inputs: kernel output [strippedbanana/sc4000-birdclef-m3-v2-adabn](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-v2-adabn)
- Key outputs/result: 4.7 CPU min per 700 files (measured).
- Kaggle description (paste-ready):

  > Runs the AdaBN-adapted baseline EfficientNet-B0 on the hidden soundscapes with the fast CPU pipeline. It scored 0.73889 public / 0.76383 private, which is 0.024 and 0.025 below the unadapted baseline (0.763 / 0.789). Inference takes 4.7 minutes for 700 files.

### sc4000-birdclef-m3-soundscape-embed

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-soundscape-embed
- Local folder: `notebooks/soundscape_embed/`
- Accelerator: GPU (NVIDIA T4). Internet on.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026). Version 1 sat queued and was re-pushed; version 2 ran.
- Submission: none
- Inputs: dataset [strippedbanana/sc4000-birdclef-m3-perch-onnx](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-perch-onnx); dataset [strippedbanana/sc4000-birdclef-m3-birdmae-base](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-birdmae-base); kernel output [jmemei/sc4000-birdclef-m1-make-mels](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-make-mels)
- Key outputs/result: 12,000 windows from 1000 train_soundscapes files, no failures; about 0.15 T4 GPU hours (version 2). Raw-variance trace: Perch 15.03, Bird-MAE 14.68.
- Kaggle description (paste-ready):

  > Computes Perch 2.0 and Bird-MAE embeddings for 12,000 five-second windows taken from 1000 unlabelled training soundscapes (field recordings similar to the test data). These are the target-domain features for the adaptation, distillation and noisy-student experiments that follow. Perch 2.0 was released after BirdCLEF 2025 closed.

### sc4000-birdclef-m3-coral-heads

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-coral-heads
- Local folder: `notebooks/coral_heads/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: none
- Inputs: kernel output [strippedbanana/sc4000-birdclef-m3-a2-perch-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-embed); kernel output [strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed); kernel output [strippedbanana/sc4000-birdclef-m3-soundscape-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-soundscape-embed); kernel output [jmemei/sc4000-birdclef-m1-make-mels](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-make-mels)
- Key outputs/result: CPU 20.7 min. First-5s val AUC, Perch: control 0.9915, mean/std 0.9915, raw CORAL 0.9917, standardised CORAL 0.9901; Bird-MAE: 0.9749, 0.9748, 0.9743, 0.9728.
- Kaggle description (paste-ready):

  > Retrains the Perch and Bird-MAE heads after aligning the clip-embedding statistics to the soundscape embeddings, using two label-free methods: mean/std matching and CORAL (matching the covariance of the two feature sets). Held-out ROC-AUC on clean clips barely moves for any variant (Perch 0.9901 to 0.9917, Bird-MAE 0.9728 to 0.9749). The heads are used by four inference notebooks whose submissions all scored lower on the leaderboard.

### sc4000-birdclef-m3-a2-coral-std-infer

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-coral-std-infer
- Local folder: `notebooks/a2_coral_std_infer/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: 58160573, public 0.68790 / private 0.70854
- Inputs: dataset [strippedbanana/sc4000-birdclef-m3-perch-onnx](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-perch-onnx); kernel output [strippedbanana/sc4000-birdclef-m3-coral-heads](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-coral-heads); kernel output [strippedbanana/sc4000-birdclef-m3-a2-perch-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-embed)
- Key outputs/result: 25.0 CPU min per 700 files (projected). Val AUC 0.9901.
- Kaggle description (paste-ready):

  > CPU inference for Perch 2.0 with a head retrained after standardised CORAL (aligning the covariance of clip embeddings to soundscape embeddings). It scored 0.68790 public / 0.70854 private, about 0.16 and 0.14 below the plain Perch head (0.84774 / 0.85226). Perch 2.0 was released after BirdCLEF 2025 closed.

### sc4000-birdclef-m3-a2-meanstd-infer

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-meanstd-infer
- Local folder: `notebooks/a2_meanstd_infer/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: 58160577, public 0.68728 / private 0.70632
- Inputs: dataset [strippedbanana/sc4000-birdclef-m3-perch-onnx](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-perch-onnx); kernel output [strippedbanana/sc4000-birdclef-m3-coral-heads](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-coral-heads); kernel output [strippedbanana/sc4000-birdclef-m3-a2-perch-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-embed)
- Key outputs/result: 21.1 CPU min per 700 files (projected). Val AUC 0.9915.
- Kaggle description (paste-ready):

  > CPU inference for Perch 2.0 with a head retrained after mean/std matching (shifting and scaling clip embeddings to the soundscape mean and spread). It scored 0.68728 public / 0.70632 private, about 0.16 and 0.15 below the plain Perch head (0.84774 / 0.85226). Perch 2.0 was released after BirdCLEF 2025 closed.

### sc4000-birdclef-m3-a3-coral-std-infer

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-coral-std-infer
- Local folder: `notebooks/a3_coral_std_infer/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: 58160580, public 0.64281 / private 0.64682
- Inputs: dataset [strippedbanana/sc4000-birdclef-m3-birdmae-base](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-birdmae-base); kernel output [strippedbanana/sc4000-birdclef-m3-coral-heads](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-coral-heads); kernel output [strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed)
- Key outputs/result: 73.4 CPU min per 700 files (projected; the closest to the 75 min budget). Val AUC 0.9728.
- Kaggle description (paste-ready):

  > CPU inference for Bird-MAE-Base with a head retrained after standardised CORAL (aligning the covariance of clip embeddings to soundscape embeddings). It scored 0.64281 public / 0.64682 private, about 0.13 below the plain Bird-MAE head (0.77640 / 0.78016) on both. The projected 73.4 minutes for 700 files was close to the time limit but it scored without a timeout.

### sc4000-birdclef-m3-a3-meanstd-infer

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-meanstd-infer
- Local folder: `notebooks/a3_meanstd_infer/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: 58160584, public 0.66971 / private 0.68033
- Inputs: dataset [strippedbanana/sc4000-birdclef-m3-birdmae-base](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-birdmae-base); kernel output [strippedbanana/sc4000-birdclef-m3-coral-heads](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-coral-heads); kernel output [strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed)
- Key outputs/result: 65.2 CPU min per 700 files (projected). Val AUC 0.9748.
- Kaggle description (paste-ready):

  > CPU inference for Bird-MAE-Base with a head retrained after mean/std matching (shifting and scaling clip embeddings to the soundscape mean and spread). It scored 0.66971 public / 0.68033 private, about 0.11 and 0.10 below the plain Bird-MAE head (0.77640 / 0.78016). It projects to about 65 minutes for 700 files.

### sc4000-birdclef-m3-coral-diag

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-coral-diag
- Local folder: `notebooks/coral_diag/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: none
- Inputs: dataset [strippedbanana/sc4000-birdclef-m3-perch-onnx](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-perch-onnx); kernel output [strippedbanana/sc4000-birdclef-m3-coral-heads](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-coral-heads); kernel output [strippedbanana/sc4000-birdclef-m3-a2-perch-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-embed); kernel output [strippedbanana/sc4000-birdclef-m3-a2-perch-infer](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-infer); kernel output [strippedbanana/sc4000-birdclef-m3-a3-mlp-head](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-mlp-head); kernel output [strippedbanana/sc4000-birdclef-m3-soundscape-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-soundscape-embed)
- Key outputs/result: CPU 3.3 min. CORAL path with the original Perch head matches the original notebook to 3.0e-8 max abs (PASS, threshold 1e-5); no pipeline bug found.
- Kaggle description (paste-ready):

  > A bug check run after the CORAL and mean/std submissions scored far below the plain heads. It compares the CORAL inference path against the original Perch notebook on 20 files (max difference 3.0e-8) and looks at prediction statistics on 12,000 soundscape windows. No pipeline bug was found; the adapted heads simply rank species differently from the original head (mean per-class Spearman correlation with it 0.29 to 0.37 for Perch). Nothing was submitted from this notebook.

### sc4000-birdclef-m3-head-seed-repeats

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-head-seed-repeats
- Local folder: `notebooks/head_seed_repeats/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: none
- Inputs: kernel output [strippedbanana/sc4000-birdclef-m3-a2-perch-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-embed); kernel output [strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed); kernel output [jmemei/sc4000-birdclef-m1-make-mels](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-make-mels)
- Key outputs/result: First-5s val AUC over seeds 42/43/44 (mean +/- std): Perch 0.99163 +/- 0.00029, Bird-MAE 0.97451 +/- 0.00076; 1211.9 s total (measured).
- Kaggle description (paste-ready):

  > Retrains the Perch and Bird-MAE heads with three random seeds (42, 43, 44) on the same fixed 80/20 split to see how much held-out ROC-AUC varies from run to run. Perch gives 0.99163 with a standard deviation of 0.00029 and Bird-MAE gives 0.97451 with 0.00076. Only the head is retrained, so this measures head-training noise, not backbone or split variation. It runs on CPU and nothing was submitted.


## 3. Distillation

### sc4000-birdclef-m3-a3-distill

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-distill
- Local folder: `notebooks/a3_distill/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: none
- Inputs: kernel output [strippedbanana/sc4000-birdclef-m3-a2-perch-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-embed); kernel output [strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed); kernel output [strippedbanana/sc4000-birdclef-m3-soundscape-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-soundscape-embed); kernel output [jmemei/sc4000-birdclef-m1-make-mels](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-make-mels)
- Key outputs/result: CPU 8.0 min. First-5s val AUC: distill 0.97504, distill+CORAL 0.97243 (original A3 0.97488).
- Kaggle description (paste-ready):

  > Trains Bird-MAE heads on labelled clips plus 12,000 unlabelled soundscape windows whose soft labels come from the Perch 2.0 head (distillation: a student learns from a teacher's predictions). Teacher probabilities are raised to the power 1.3 and values under 0.01 set to zero, with soundscape loss weight 0.5. It produces two heads, plain and CORAL-aligned, with held-out ROC-AUC 0.97504 and 0.97243. Perch 2.0 was released after BirdCLEF 2025 closed.

### sc4000-birdclef-m3-a3-distill-infer

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-distill-infer
- Local folder: `notebooks/a3_distill_infer/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: 58161782, public 0.81238 / private 0.80208
- Inputs: dataset [strippedbanana/sc4000-birdclef-m3-birdmae-base](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-birdmae-base); kernel output [strippedbanana/sc4000-birdclef-m3-a3-distill](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-distill); kernel output [strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed)
- Key outputs/result: 67.5 CPU min per 700 files (projected).
- Kaggle description (paste-ready):

  > CPU inference for Bird-MAE-Base with the head distilled from Perch 2.0 predictions on soundscape windows. It scored 0.81238 public / 0.80208 private, up 0.036 and 0.022 from the plain Bird-MAE head (0.77640 / 0.78016). The projected run time is 67.5 minutes for 700 files. Perch 2.0 was released after BirdCLEF 2025 closed.

### sc4000-birdclef-m3-a3-distill-coral-infer

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-distill-coral-infer
- Local folder: `notebooks/a3_distill_coral_infer/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: 58162938, public 0.69343 / private 0.68854
- Inputs: dataset [strippedbanana/sc4000-birdclef-m3-birdmae-base](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-birdmae-base); kernel output [strippedbanana/sc4000-birdclef-m3-a3-distill](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-distill); kernel output [strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed)
- Key outputs/result: 61.9 CPU min per 700 files (projected). Val AUC 0.9724.
- Kaggle description (paste-ready):

  > CPU inference for Bird-MAE-Base with the Perch-distilled head after an extra CORAL alignment of the clip embeddings. It scored 0.69343 public / 0.68854 private, about 0.12 below the distilled head without CORAL (0.81238 / 0.80208). It was submitted by a rule fixed beforehand, whatever the score. Perch 2.0 was released after BirdCLEF 2025 closed.

### sc4000-birdclef-m3-a2-a1distill

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-a1distill
- Local folder: `notebooks/a2_a1distill/`
- Accelerator: GPU (NVIDIA T4). Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: none
- Inputs: kernel output [strippedbanana/sc4000-birdclef-m3-a1-v2s-train](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-v2s-train); kernel output [strippedbanana/sc4000-birdclef-m3-a2-perch-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-embed); kernel output [strippedbanana/sc4000-birdclef-m3-soundscape-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-soundscape-embed); kernel output [jmemei/sc4000-birdclef-m1-make-mels](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-make-mels)
- Key outputs/result: T4, 4.5 min, about 0.08 GPU hours (measured). First-5s val AUC 0.99182 (recorded, not used for selection); teacher labels 3.7% non-zero, 1378 windows with a label >= 0.5.
- Kaggle description (paste-ready):

  > Trains the Perch 2.0 head again, this time also on 12,000 unlabelled soundscape windows whose soft labels come from the A1 EfficientNetV2-S model (distillation: one model learns from another's predictions). Labels are the teacher probabilities raised to the 1.3 power, with values under 0.01 set to zero. Held-out ROC-AUC on clean clips is 0.99182, essentially unchanged, but the leaderboard score improved. Perch 2.0 was released after BirdCLEF 2025 closed.

### sc4000-birdclef-m3-a2-a1distill-infer

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-a1distill-infer
- Local folder: `notebooks/a2_a1distill_infer/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: 58164973, public 0.86651 / private 0.85737
- Inputs: dataset [strippedbanana/sc4000-birdclef-m3-perch-onnx](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-perch-onnx); kernel output [strippedbanana/sc4000-birdclef-m3-a2-a1distill](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-a1distill); kernel output [strippedbanana/sc4000-birdclef-m3-a2-perch-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-embed)
- Key outputs/result: 26.5 CPU min per 700 files (projected).
- Kaggle description (paste-ready):

  > CPU inference for Perch 2.0 with the head distilled from A1 predictions on soundscape windows. It scored 0.86651 public / 0.85737 private, up 0.019 and 0.005 from the plain Perch head (0.84774 / 0.85226), and was the best single model of the project. The projected run time is 26.5 minutes for 700 files. Perch 2.0 was released after BirdCLEF 2025 closed.


## 4. Noisy student

### sc4000-birdclef-m3-ns-perch-all

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-perch-all
- Local folder: `notebooks/ns_perch_all/`
- Accelerator: GPU (NVIDIA T4). Internet on.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: none
- Inputs: dataset [strippedbanana/sc4000-birdclef-m3-perch-onnx](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-perch-onnx); kernel output [strippedbanana/sc4000-birdclef-m3-soundscape-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-soundscape-embed)
- Key outputs/result: 9726 files, 116,712 windows (12 per file), 0 failed; 0.91 T4 GPU hours; notebook wall 3277 s (measured).
- Kaggle description (paste-ready):

  > Embeds every 5 s window of all 9726 training soundscapes (116,712 windows) with Perch 2.0 on the GPU, as the first teacher input for the noisy-student step. A check against the earlier 12,000-window run differs by at most 3.4e-7. Perch 2.0 was released after BirdCLEF 2025 closed.

### sc4000-birdclef-m3-ns-a1-all

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-a1-all
- Local folder: `notebooks/ns_a1_all/`
- Accelerator: GPU (NVIDIA T4). Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: none
- Inputs: kernel output [strippedbanana/sc4000-birdclef-m3-a1-v2s-train](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-v2s-train); kernel output [strippedbanana/sc4000-birdclef-m3-a2-a1distill](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-a1distill); kernel output [strippedbanana/sc4000-birdclef-m3-soundscape-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-soundscape-embed)
- Key outputs/result: Same 116,712 windows, 0 failed; 0.27 T4 GPU hours; notebook wall 990 s (measured).
- Kaggle description (paste-ready):

  > Runs the unadapted A1 EfficientNetV2-S model on the same 116,712 five-second windows of all training soundscapes and saves its per-species probabilities. These are the second teacher input for the noisy-student step. The output matches the earlier 12,000-window run to within 5.4e-7.

### sc4000-birdclef-m3-ns-pseudo-labels

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-pseudo-labels
- Local folder: `notebooks/ns_pseudo_labels/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: none
- Inputs: kernel output [strippedbanana/sc4000-birdclef-m3-ns-perch-all](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-perch-all); kernel output [strippedbanana/sc4000-birdclef-m3-ns-a1-all](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-a1-all); kernel output [strippedbanana/sc4000-birdclef-m3-a2-perch-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-embed); kernel output [strippedbanana/sc4000-birdclef-m3-a2-a1distill](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-a1distill)
- Key outputs/result: CPU 2.1 min. MAIN labels 2.50% non-zero, 11,430 windows with a label >= 0.5; ABLATION 0.74%, 7,952 windows.
- Kaggle description (paste-ready):

  > Turns the Perch and A1 predictions on all training soundscape windows into pseudo-labels (soft labels guessed by a teacher). MAIN is the mean of A1 and the original Perch head, and ABLATION is the Perch head distilled from A1 alone; both are raised to the power 1.3 and values below 0.01 are set to zero. MAIN ends up 2.50% non-zero and ABLATION 0.74%. Perch 2.0 was released after BirdCLEF 2025 closed.

### sc4000-birdclef-m3-ns-mel-check

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-mel-check
- Local folder: `notebooks/ns_mel_check/`
- Accelerator: GPU (NVIDIA T4). Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: none
- Inputs: kernel output [strippedbanana/sc4000-birdclef-m3-a1-v2s-train](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-v2s-train); kernel output [jmemei/sc4000-birdclef-m1-make-mels](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-make-mels)
- Key outputs/result: T4, 2.0 min, about 0.05 GPU hours. Max abs difference 1.3e-4 on model input, 6e-6 on probabilities, 0 argmax changes (50 clips + 50 windows).
- Kaggle description (paste-ready):

  > Checks that computing mel spectrograms on the GPU with torchaudio gives the same input to the A1 model as the librosa CPU path. On 50 training clips and 50 soundscape windows the model input differs by at most 1.3e-4 and the predicted probabilities by at most 6e-6, with no change in the top prediction. This allowed the student networks to compute their spectrograms on the fly on the GPU.

### sc4000-birdclef-m3-ns-student-main

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-main
- Local folder: `notebooks/ns_student_main/`
- Accelerator: GPU (NVIDIA T4). Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: none
- Inputs: dataset [vladimirsydor/bird-clef-2025-all-pretrained-models](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-all-pretrained-models); kernel output [jmemei/sc4000-birdclef-m1-make-mels](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-make-mels); kernel output [strippedbanana/sc4000-birdclef-m3-ns-pseudo-labels](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-pseudo-labels)
- Key outputs/result: Best val AUC 0.98650 (epoch 14), 15 epochs, 4.7 min/epoch; about 1.18 T4 GPU hours (measured).
- Kaggle description (paste-ready):

  > Noisy-student training (a new network learns from labelled clips plus pseudo-labelled soundscape windows, with noise added): an EfficientNetV2-S starting from the bird-pretrained weights, trained on labelled clips and 30,000 soundscape windows per epoch labelled by the MAIN teacher (mean of A1 and Perch). Held-out ROC-AUC reaches 0.98650. Its submission scored 0.85985 public / 0.85185 private. Perch 2.0 was released after BirdCLEF 2025 closed.

### sc4000-birdclef-m3-ns-student-ablation

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-ablation
- Local folder: `notebooks/ns_student_ablation/`
- Accelerator: GPU (NVIDIA T4). Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: none
- Inputs: dataset [vladimirsydor/bird-clef-2025-all-pretrained-models](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-all-pretrained-models); kernel output [jmemei/sc4000-birdclef-m1-make-mels](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-make-mels); kernel output [strippedbanana/sc4000-birdclef-m3-ns-pseudo-labels](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-pseudo-labels)
- Key outputs/result: Best val AUC 0.98541 (epoch 14), 15 epochs, 4.4 min/epoch; about 1.10 T4 GPU hours (measured).
- Kaggle description (paste-ready):

  > The comparison run for the noisy student: same network, recipe and soundscape windows, but labelled by the ABLATION teacher (the Perch head distilled from A1, with sparser labels). Held-out ROC-AUC reaches 0.98541. Its submission scored 0.84003 public / 0.83794 private. Because the teacher and the label density both differ from MAIN, the gap to MAIN cannot be put down to the teacher alone. Perch 2.0 was released after BirdCLEF 2025 closed.

### sc4000-birdclef-m3-ns-student-main-infer

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-main-infer
- Local folder: `notebooks/ns_student_main_infer/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: 58181262, public 0.85985 / private 0.85185
- Inputs: kernel output [strippedbanana/sc4000-birdclef-m3-ns-student-main](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-main)
- Key outputs/result: 9.9 CPU min per 700 files (measured).
- Kaggle description (paste-ready):

  > CPU inference for the MAIN noisy-student EfficientNetV2-S with the fast pipeline. It scored 0.85985 public / 0.85185 private, up 0.032 and 0.035 from the A1 model it learned from (0.82822 / 0.81642). A run for 700 files takes about 9.9 minutes.

### sc4000-birdclef-m3-ns-student-ablation-infer

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-ablation-infer
- Local folder: `notebooks/ns_student_ablation_infer/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: 58181267, public 0.84003 / private 0.83794
- Inputs: kernel output [strippedbanana/sc4000-birdclef-m3-ns-student-ablation](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-ablation)
- Key outputs/result: 10.0 CPU min per 700 files (measured).
- Kaggle description (paste-ready):

  > CPU inference for the ABLATION noisy-student EfficientNetV2-S with the fast pipeline. It scored 0.84003 public / 0.83794 private, up 0.012 and 0.022 from A1 but below the MAIN student. A run for 700 files takes about 10.0 minutes.

### sc4000-birdclef-m3-ns-head-seeds

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-head-seeds
- Local folder: `notebooks/ns_head_seeds/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: none
- Inputs: kernel output [strippedbanana/sc4000-birdclef-m3-a2-perch-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-embed); kernel output [strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed); kernel output [strippedbanana/sc4000-birdclef-m3-soundscape-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-soundscape-embed); kernel output [strippedbanana/sc4000-birdclef-m3-a2-a1distill](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-a1distill); kernel output [jmemei/sc4000-birdclef-m1-make-mels](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-make-mels)
- Key outputs/result: First-5s val AUC over seeds 42/43/44: Perch head distilled from A1 0.99133 +/- 0.00047; Bird-MAE Perch-distilled head 0.97506 +/- 0.00012. Run time about 22.4 min on CPU (last log timestamp 1,341 s, measured).
- Kaggle description (paste-ready):

  > Repeats the distilled Perch and Bird-MAE heads with three random seeds (42, 43, 44) on the same split to measure seed noise. The A1-distilled Perch head gives 0.99133 with a standard deviation of 0.00047, and the Perch-distilled Bird-MAE head gives 0.97506 with 0.00012. It runs on CPU and nothing was submitted.


## 5. Ensemble

### sc4000-birdclef-m3-ens-e1

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ens-e1
- Local folder: `notebooks/ens_e1/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: 58196484, public 0.88352 / private 0.87203
- Inputs: dataset [strippedbanana/sc4000-birdclef-m3-perch-onnx](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-perch-onnx); kernel output [strippedbanana/sc4000-birdclef-m3-ns-student-main](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-main); kernel output [strippedbanana/sc4000-birdclef-m3-ns-student-ablation](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-ablation); kernel output [strippedbanana/sc4000-birdclef-m3-a2-a1distill](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-a1distill); kernel output [strippedbanana/sc4000-birdclef-m3-a2-perch-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-embed)
- Key outputs/result: 28.3 CPU min per 700 files (measured on a 700-file dry run; STATE.md calls it projected).
- Kaggle description (paste-ready):

  > Final-ensemble variant E1: per-class rank average (each species' scores are replaced by their ranks, then averaged) of the MAIN noisy student and the A1-distilled Perch head, equal weights, no smoothing. Audio is decoded once and shared. It scored 0.88352 public / 0.87203 private. Perch 2.0 was released after BirdCLEF 2025 closed.

### sc4000-birdclef-m3-ens-e2

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ens-e2
- Local folder: `notebooks/ens_e2/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: 58196486, public 0.88662 / private 0.88008
- Inputs: dataset [strippedbanana/sc4000-birdclef-m3-perch-onnx](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-perch-onnx); kernel output [strippedbanana/sc4000-birdclef-m3-ns-student-main](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-main); kernel output [strippedbanana/sc4000-birdclef-m3-ns-student-ablation](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-ablation); kernel output [strippedbanana/sc4000-birdclef-m3-a2-a1distill](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-a1distill); kernel output [strippedbanana/sc4000-birdclef-m3-a2-perch-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-embed)
- Key outputs/result: 30.4 CPU min per 700 files (measured on a 700-file dry run; STATE.md calls it projected).
- Kaggle description (paste-ready):

  > Final-ensemble variant E2: E1 plus temporal smoothing, where each 5 s window's score is mixed with its neighbours in the same file (weights 0.25, 0.5, 0.25) before ranking. It scored 0.88662 public / 0.88008 private, up 0.003 and 0.008 from E1. Perch 2.0 was released after BirdCLEF 2025 closed.

### sc4000-birdclef-m3-ens-e3

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ens-e3
- Local folder: `notebooks/ens_e3/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: 58198886, public 0.88058 / private 0.87522
- Inputs: dataset [strippedbanana/sc4000-birdclef-m3-perch-onnx](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-perch-onnx); kernel output [strippedbanana/sc4000-birdclef-m3-ns-student-main](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-main); kernel output [strippedbanana/sc4000-birdclef-m3-ns-student-ablation](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-ablation); kernel output [strippedbanana/sc4000-birdclef-m3-a2-a1distill](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-a1distill); kernel output [strippedbanana/sc4000-birdclef-m3-a2-perch-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-embed)
- Key outputs/result: 39.9 CPU min per 700 files (measured on a 700-file dry run; STATE.md calls it projected).
- Kaggle description (paste-ready):

  > Final-ensemble variant E3: E2 with the ABLATION noisy student added, so the MAIN student, the ABLATION student and the A1-distilled Perch head each get one third. It scored 0.88058 public / 0.87522 private, lower than E2 on both. Perch 2.0 was released after BirdCLEF 2025 closed.

### sc4000-birdclef-m3-ens-e4

- Link: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ens-e4
- Local folder: `notebooks/ens_e4/`
- Accelerator: CPU only. Internet off.
- Visibility: Private
- Status: COMPLETE (verified 10 Nov 2026)
- Submission: 58198889, public 0.88749 / private 0.88029
- Inputs: dataset [strippedbanana/sc4000-birdclef-m3-perch-onnx](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-perch-onnx); kernel output [strippedbanana/sc4000-birdclef-m3-ns-student-main](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-main); kernel output [strippedbanana/sc4000-birdclef-m3-ns-student-ablation](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-ablation); kernel output [strippedbanana/sc4000-birdclef-m3-a2-a1distill](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-a1distill); kernel output [strippedbanana/sc4000-birdclef-m3-a2-perch-embed](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-embed)
- Key outputs/result: 32.3 CPU min per 700 files (measured on a 700-file dry run; STATE.md calls it projected). Selected as the final model by public score.
- Kaggle description (paste-ready):

  > Final-ensemble variant E4: the same as E2 but weighted 0.4 for the MAIN noisy student and 0.6 for the A1-distilled Perch head. It scored 0.88749 public / 0.88029 private, the best of the project and the selected final model, though only 0.001 above E2 on public, which is within noise. Perch 2.0 was released after BirdCLEF 2025 closed.

## Private datasets

Both datasets belong to `strippedbanana` and are private.

### sc4000-birdclef-m3-perch-onnx

- Link: https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-perch-onnx
- Contents: `perch_v2.onnx` (409 MB) and a LICENSE file. The SHA-256 of the ONNX file is checked by the notebooks that load it.
- Licence note: a community ONNX conversion of Perch 2.0, not an official release; the LICENSE file is included. Perch 2.0 was released after BirdCLEF 2025 closed.
- Used by: soundscape-embed, ns-perch-all, the Perch inference notebooks (A2 and variants, coral-diag) and the four ensemble notebooks.

### sc4000-birdclef-m3-birdmae-base

- Link: https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-birdmae-base
- Contents: `model.safetensors` (342 MB), `config.json`, `preprocessor_config.json`, `configuration_bird_mae.py`, `feature_extractor.py`, `modeling_bird_mae.py`, copied from `DBD-research-group/Bird-MAE-Base` at commit `6cc416d1`; SHA-256 verified.
- Licence note: the Bird-MAE weights declare no licence on the Hugging Face card, GitHub repository or arXiv listing, so this dataset stays private.
- Used by: soundscape-embed and the Bird-MAE inference notebooks (A3 and variants).

## Upstream inputs not owned by this account

- [jmemei/sc4000-birdclef-m1-make-mels](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-make-mels): our team's baseline notebook that caches the log-mel spectrograms and the train/validation split used by every training notebook here.
- [jmemei/sc4000-birdclef-m1-train-v2](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-train-v2): our team's trained baseline v2 (EfficientNet-B0), used as the baseline for timing and for the v2 AdaBN experiment.
- [vladimirsydor/bird-clef-2025-all-pretrained-models](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-all-pretrained-models): bird-pretrained backbone weights from the BirdCLEF 2025 2nd-place solution, used to initialise A1 and the noisy students.
