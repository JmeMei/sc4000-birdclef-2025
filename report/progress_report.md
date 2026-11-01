# BirdCLEF+ 2025: Identifying Species from Rainforest Soundscapes

*SC4000 Machine Learning — Group project report (DRAFT, findings as of 1 Nov 2026)*

> Draft text for the AAAI template in `report/AAAI-word-template.docx`. Sections marked **TODO**
> are filled in once later experiments finish. Figures live in `report/figures/`.

---

## Abstract

We tackle the BirdCLEF+ 2025 Kaggle competition: given 1-minute audio recordings from
El Silencio Natural Reserve (Colombia), predict which of 206 species (birds, amphibians,
mammals and insects) are calling in each 5-second window. Our exploratory analysis shows three
core difficulties: heavy class imbalance, weak (file-level) labels, and a domain shift between
clean, focused training recordings and noisy passive-monitoring soundscapes at test time.
As a baseline we convert audio to log-mel spectrograms and fine-tune an ImageNet-pretrained
EfficientNet-B0, reaching a macro ROC-AUC of **0.973** on a held-out 20% of the training clips.
**TODO:** public/private leaderboard score, and results of the SED model, focal loss, and
pseudo-labelling experiments.

## 1. Introduction

Passive acoustic monitoring lets ecologists survey biodiversity at scale by leaving recorders
in the field, but the resulting hours of audio are too much to label by hand. The task is to
build a model that, for each 5-second window of a soundscape, outputs a probability for every
one of 206 species. The competition metric is the macro-averaged ROC-AUC across species
(species with no positive window in the test set are skipped), so every species counts
equally regardless of how common it is.

It is a *code competition*: the submission is a Kaggle notebook that must run on **CPU only**,
with internet disabled, within **90 minutes**. This constraint shapes the model choices below.

## 2. Data and exploratory analysis

| Statistic | Value |
|---|---|
| Labelled training clips (`train_audio`) | 28,564 |
| Species (taxonomy) | 206 (Aves ≈146, Amphibia ≈34, Insecta ≈17, Mammalia ≈9) |
| Species with no training clip | 0 |
| Species with ≤ 5 clips / ≤ 20 clips | 30 / 55 |
| Clips per species: min / median / max | 2 / 80 / 990 |
| Clips with secondary labels | 9.3% (mean 1.48 extra labels when present) |
| Median clip duration | 20.5 s (8.1% shorter than 5 s) |
| Audio format | 32 kHz mono `.ogg` |
| Unlabelled 1-min soundscapes (`train_soundscapes`) | 9,726 |
| Clips with quality rating 0 (unrated) | ≈ 8,000 |

**Finding 1: severe class imbalance** (Fig. `class_imbalance.png`, `taxonomic_groups.png`).
The most common species has 990 clips, while 30 species have five or fewer. Birds dominate
both the species list and, even more, the clip count; insects and mammals together have only a
few hundred clips. Because the metric is macro-averaged, the rare classes weigh as much as
the common ones, so methods that help rare classes (focal loss, class-balanced sampling,
pseudo-labels) should pay off.

**Finding 2: weak labels** (Fig. `durations.png`). Each training file carries one label for the
whole recording, with a median length of about 20 s, but the animal may only call for a few
seconds of it. Training on a fixed 5 s crop (as in our baseline) can therefore show the model
windows that contain only background noise but are labelled as a species. Secondary labels
are present in only 9.3% of clips and are likely incomplete.

**Finding 3: domain shift** (Fig. `example_spectrograms.png`, `soundscape_windows.png`).
Training clips are mostly focused recordings from Xeno-canto / iNaturalist, where the target
animal is loud and close. Test soundscapes are continuous 60 s recordings from fixed recorders
with rain, insects, overlapping calls and distant sources. The 9,726 unlabelled soundscapes
come from the same recorders as the test set, which makes them valuable for semi-supervised
learning (pseudo-labelling).

**Finding 4: variable label quality** (Fig. `ratings.png`). Most rated clips score 4–5, but
roughly 8,000 clips are unrated (rating 0), so quality filtering has to be done with care.

## 3. Method

### 3.1 Baseline: EfficientNet-B0 on log-mel spectrograms

- **Input:** the first 5 s of each clip (zero-padded if shorter), resampled at 32 kHz, converted
  to a 128-band log-mel spectrogram (n_fft 2048, hop 512, 50–14,000 Hz), scaled to [0, 1].
  Spectrograms are computed once and cached in memory (float16, ~2.3 GB).
- **Model:** `efficientnet_b0` from `timm`, ImageNet-pretrained, adapted to 1 input channel,
  206 sigmoid outputs (multi-label).
- **Training:** primary label only, binary cross-entropy loss, AdamW (lr 1e-3, weight decay
  1e-2), one-cycle schedule, batch size 64, 15 epochs, mixed precision on a Kaggle T4 GPU
  (~57 s per epoch). No data augmentation.
- **Validation:** stratified 80/20 hold-out split of the training clips; macro ROC-AUC over
  species present in the hold-out set, mirroring the competition metric.
- **Inference:** each test soundscape is cut into twelve 5 s windows, converted with the same
  mel settings, and scored on CPU. On a dry run over 20 unlabelled soundscapes (240 windows),
  spectrogram extraction took 33 s and the forward pass 8 s with 4 CPU cores.

### 3.2 Planned improvements (TODO: fill in as completed)

1. **Focal loss** in place of BCE, as an ablation targeting class imbalance.
2. **Sound-event-detection (SED) model on 20 s chunks**: predicts frame-level presence and
   pools it into a clip score, so the model learns *where* in a clip the species calls. This
   directly addresses the weak-label problem (Finding 2). Re-implemented from published
   write-ups and cited accordingly.
3. **Pseudo-labelling (one round of Noisy Student)** on the unlabelled soundscapes to reduce
   the domain shift (Finding 3).
4. **Ensemble** of the best models, subject to the 90-minute CPU budget.

## 4. Results

### 4.1 Baseline training curve

| Epoch | Train loss | Val loss | Val macro ROC-AUC |
|---|---|---|---|
| 1 | 0.1364 | 0.0278 | 0.592 |
| 2 | 0.0219 | 0.0180 | 0.903 |
| 4 | 0.0108 | 0.0120 | 0.967 |
| **7** | **0.0047** | **0.0101** | **0.973** (best, saved) |
| 10 | 0.0007 | 0.0110 | 0.963 |
| 15 | 0.0001 | 0.0115 | 0.960 |

(Fig. `baseline_training.png`.) Validation AUC peaks at epoch 7 and then declines while
training loss keeps falling towards zero, a clear sign of **overfitting**: with no
augmentation and a fixed crop, the model memorises the training spectrograms. Augmentation
(mixup, time/frequency masking, random crops) and fewer epochs are obvious next steps.

### 4.2 Experiment summary

| Experiment | Val macro AUC | Public LB | Notes |
|---|---|---|---|
| EfficientNet-B0, BCE, first 5 s, no aug | 0.973 | TODO | Baseline |
| + Focal loss | TODO | TODO | Ablation |
| SED on 20 s chunks | TODO | TODO | |
| + 1 round pseudo-labels | TODO | TODO | |
| Ensemble | TODO | TODO | |

**Caution on the validation score.** Our hold-out clips come from the same clean,
focused-recording distribution as training, whereas the test set is noisy soundscapes.
A validation AUC of 0.97 will therefore overstate the leaderboard score; the leaderboard is the
more honest measure of generalisation to soundscapes.

## 5. Discussion

**TODO** after further experiments: which change helped most, how rare-class AUC compares to
common-class AUC, the inference-time budget of each model, and what we would try with more
time (e.g. larger audio backbones such as Bird-MAE, more pseudo-labelling rounds).

## 6. Conclusion

**TODO.**

## Acknowledgements and references

- BirdCLEF+ 2025 competition and data: Kaggle / Cornell Lab of Ornithology / LifeCLEF.
- `timm` (PyTorch Image Models), EfficientNet (Tan & Le, 2019), librosa.
- **TODO:** cite any Kaggle solution write-ups or public notebooks consulted (required by the
  course; using other participants' code counts as plagiarism, so re-implement from write-ups
  only).

## Team contributions

**TODO:** one line per member.
