# Pretrained models, domain transfer and inference speed (with the final ensemble)

## Setup

Every experiment follows the pattern "v2 + X". Our team's baseline v2 recipe, its stratified 80/20 split (seed 42) and its inference protocol stay fixed, and only one component changes. Models are scored as late submissions to BirdCLEF+ 2025, and the public and private macro ROC-AUC are both reported.

Two lessons from our team's baseline work apply throughout. Local validation AUC does not predict the leaderboard, so we select by **public** LB, and public differences below 0.01 are treated as noise.

We pre-registered every variant in the later phases, with its selection rule, before seeing any of its scores. All submissions run on CPU with internet off and must project ≤ 75 min for 700 one-minute soundscapes; the hard limit is 90.

**Pretrained models.**
- **A1:** EfficientNetV2-S with a bird-pretrained backbone from the 2025 2nd-place solution (Sydorskyi & Gonçalves). Only the pretrained backbone is used, not their trained models. It is fine-tuned with v2's recipe on v2's mel features, using a per-window z-score/min-max input.
- **A2:** frozen Perch 2.0 1536-d embeddings (ONNX on CPU) with an MLP head (hidden 512, dropout 0.3).
- **A3:** frozen Bird-MAE-Base 768-d embeddings with an MLP head.

**Focal-to-soundscape transfer.** Training audio is focal recordings, but the test set is passive soundscapes. Using unlabelled `train_soundscapes` only, we tried three things. The first is label-free statistics alignment (AdaBN, CORAL, and mean/std matching). The second is cross-model distillation, where soft labels from one model on soundscape windows are used to train another model's head. The third is one round of noisy-student training: a fresh A1-architecture student is trained on focal clips plus 116,712 soundscape windows labelled by a teacher. The teacher is mean(A1, A2), sharpened with `p^1.3`, with values < 0.01 set to 0.

**Final ensemble.** In one CPU notebook, audio is decoded once and shared between models. Each model's per-class probabilities are ranked over all test rows. Optionally, a 0.25/0.5/0.25 temporal smoothing is first applied within each file. The final prediction is a weighted mean of the ranks.

## Results

| Model | Val AUC | Public | Private | CPU min / 700 |
|---|---|---|---|---|
| v2 baseline (team baseline, B0 ImageNet) | – | 0.763 | 0.789 | 5.5 (fast pipeline) |
| A1 V2-S, bird-pretrained | 0.9832 | 0.82822 | 0.81642 | 8.0 (fast pipeline) |
| A2 Perch 2.0 + MLP | 0.9916 | 0.84774 | 0.85226 | 25.0 |
| A3 Bird-MAE-Base + MLP | 0.9749 | 0.77640 | 0.78016 | 61.6 |
| A3 + Perch-distilled head | 0.9750 | 0.81238 | 0.80208 | 67.5 |
| A2 + A1-distilled head | 0.9918 | 0.86651 | 0.85737 | 26.5 |
| Noisy student, MAIN teacher mean(A1, A2) | 0.9865 | 0.85985 | 0.85185 | 9.9 |
| Noisy student, ABLATION teacher (A2 + A1-distilled) | 0.9854 | 0.84003 | 0.83794 | 10.0 |
| **Ensemble E4 (final)** | – | **0.88749** | **0.88029** | 32.3 |

Statistics alignment hurt every model. Δ public / private is against each unadapted model:

| Method | Model | Public | Private | Δ public / private |
|---|---|---|---|---|
| AdaBN | A1 | 0.77210 | 0.76573 | −0.056 / −0.051 |
| AdaBN | v2 | 0.73889 | 0.76383 | −0.024 / −0.025 |
| Standardised CORAL | A2 | 0.68790 | 0.70854 | −0.160 / −0.144 |
| Standardised CORAL | A3 | 0.64281 | 0.64682 | −0.134 / −0.133 |
| Mean/std matching | A2 | 0.68728 | 0.70632 | −0.161 / −0.146 |
| Mean/std matching | A3 | 0.66971 | 0.68033 | −0.107 / −0.100 |

Ensemble variants (pre-registered; Δ is against the best single model, A2 + A1-distilled):

| Variant | Models (weights) | Smoothing | Public | Private | Δ public / private |
|---|---|---|---|---|---|
| E1 | MAIN 0.5 + A2d 0.5 | no | 0.88352 | 0.87203 | +0.0170 / +0.0147 |
| E2 | MAIN 0.5 + A2d 0.5 | yes | 0.88662 | 0.88008 | +0.0201 / +0.0227 |
| E3 | MAIN + ABLATION + A2d, 1/3 each | yes | 0.88058 | 0.87522 | +0.0141 / +0.0179 |
| **E4** | **MAIN 0.4 + A2d 0.6** | **yes** | **0.88749** | **0.88029** | **+0.0210 / +0.0229** |

We also repeated head training on CPU with seeds 42/43/44 on the fixed split. Val AUC (mean ± std) is 0.99133 ± 0.00047 for A2 + A1-distilled and 0.97506 ± 0.00012 for A3 Perch-distilled. Seed noise on validation is therefore at least ten times smaller than the leaderboard noise level.

**Inference speed.** File-level multiprocessing (4 workers × 1 thread), one mel call per file, and batching the 12 windows per file cut v2's whole notebook from 11.1 to 5.5 CPU min on 700 `train_soundscapes`. A1 in the same pipeline takes 8.0 min. Details are in `report/member3-4_pretrained_ensemble/results/timing_table.md`.

## Discussion

- **Domain-pretrained models give the largest single gains.** Bird-pretrained A1 adds +0.065 public over v2, and frozen Perch 2.0 with a small head (A2) adds +0.085 public / +0.063 private. Bird-MAE (A3) has a val AUC close to Perch's but transfers poorly to soundscapes (+0.013 public, −0.009 private). This is another case where validation AUC does not predict the leaderboard.
- **Label-free statistics alignment failed everywhere,** by 0.02–0.16. A diagnostic rejected our hypothesis that the adapted heads become over-confident on background windows. Instead, the adapted heads reorder windows within each class: per-class Spearman against the original head is 0.29–0.37 for Perch and 0.57–0.72 for Bird-MAE.
- **Cross-model distillation on unlabelled soundscapes worked** where alignment did not. Distilling A2 into A3's head gave +0.036 / +0.022, and distilling A1 into A2's head gave +0.019 / +0.005, making it the best single model.
- **Noisy student** raised the A1 architecture by +0.032 / +0.035 and narrowed the val-to-public gap from 0.155 to 0.127. The MAIN teacher beat the ABLATION teacher by 0.020 public. However, the two pseudo-label sets also differ in density (2.50% vs 0.74% non-zero), so this gap cannot be credited to the teacher alone.
- **The ensemble of a spectrogram CNN and a Perch head gave the largest gain** (+0.017 / +0.015 for E1), showing the two model families are complementary. Smoothing added a further +0.003 public / +0.008 private. Adding the second student (E3) hurt, because it moved weight from Perch to two highly correlated CNNs. E4 was selected by the pre-registered rule, but E4 and E2 differ by under 0.001, which is within noise.
- Overall, E4 is +0.124 public / +0.091 private over v2 and well above the 0.810 team goal.

## Limitations

- A1 is a backbone change from v2's B0, not a weights-only swap. Its mel normalisation only partly matches the backbone's pretraining.
- Perch 2.0 was released after the competition closed, and it was trained on Xeno-canto and iNaturalist. Overlap with `train_audio` may inflate A2's validation AUC. The ONNX model is a community conversion.
- Bird-MAE declares no weights licence, so its dataset stays private.
- Selection uses the public leaderboard, a small test subset. Several comparisons (E4 vs E2, A2d vs the MAIN student) are within the 0.01 public noise level. Private scores are reported only for transparency and were never used to choose.
- Each phase scored only a few pre-registered variants, so the ensemble weights are not tuned. A wider weight search on the public LB would risk overfitting it.
- Noisy student was run for only one round with a single seed. The teacher and pseudo-label density were confounded.
- CPU times come from 700-file dry runs on `train_soundscapes` in Kaggle CPU sessions; the CLI does not report the scored runtime. ONNX/OpenVINO export of the CNNs was not completed.
