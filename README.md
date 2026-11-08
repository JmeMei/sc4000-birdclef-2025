# SC4000 Course Project — BirdCLEF+ 2025

> **Private repo.** The course rules say not to release project code publicly. Keep this repository private.

Group project for SC4000/CZ4041/CE4041 Machine Learning (NTU), on the Kaggle competition
[BirdCLEF+ 2025](https://www.kaggle.com/competitions/birdclef-2025/).

**Task:** identify which of 206 species (birds, amphibians, mammals, insects) are calling in each
5-second window of 1-minute soundscape recordings from El Silencio Natural Reserve, Colombia.

**Metric:** macro-averaged ROC-AUC (classes with no true positives in the test set are skipped).

**Submission:** code competition — a Kaggle notebook that runs on **CPU only**, internet off, within
the time limit.

**Deadline:** report + video link + submission file via NTULearn by **13 Nov 2026, 11:59pm**.

## Layout

```
configs/       experiment configs (yaml)
data/          competition data (gitignored — download with scripts/download_data.sh)
models/        trained weights (gitignored)
notebooks/     EDA, experiments, Kaggle submission notebook
report/        AAAI report source + figures
scripts/       helper scripts
src/           reusable code (dataset, features, models, training)
submissions/   generated submission CSVs (gitignored)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh
```

## Running on Kaggle (no local data or GPU needed)

Write code in VS Code, then run it on Kaggle where the data is already attached.
Each notebook lives in its own folder under `notebooks/` with a `kernel-metadata.json`.

```bash
source .venv/bin/activate
bash scripts/kaggle_run.sh notebooks/eda
```

This pushes the notebook as a **private** Kaggle kernel, waits for it to finish, and
downloads the outputs (figures, CSVs, models) to `outputs/eda/` (gitignored).

To make a new notebook: copy `notebooks/eda/kernel-metadata.json` into a new folder, then change
`id`, `title` and `code_file`. Set `"enable_gpu": true` for training. Kaggle gives about 30 GPU hours a week.
The script refuses to push if `"is_private"` is not `true`.

Requires the Kaggle CLI token at `~/.kaggle/access_token`.

## Experiment log

Record every leaderboard submission here — the report needs baselines and ablations.

| Date | Who | Experiment | Local CV AUC | Public LB | Notes |
|------|-----|------------|--------------|-----------|-------|
| 2026-11-01 | Jamie | Baseline: EfficientNet-B0 (ImageNet), first 5 s log-mel, primary label only, BCE, no augmentation, 15 epochs (best = epoch 7) | 0.9732 | 0.71681 (private 0.69908) | Rank ~1600/2032 (top 79%). Overfits after epoch 7; large gap between local CV and LB because training clips differ from soundscapes |
| 2026-11-07 | Jamie | Ablation `crop`: baseline + random 5 s crops from whole clip (m1-train-crop v1, m1-infer-crop v1) | 0.9818 (epoch 15) | 0.75169 (private 0.75384) | Δ vs baseline LB: +0.0349. No overfitting; still improving at epoch 15 |
| 2026-11-07 | Jamie | Ablation `noise`: baseline + soundscape noise (p 0.5, −20 to −3 dB) + mixup (p 0.5) (m1-train-noise v1, m1-infer-noise v1) | 0.9748 (epoch 14) | 0.73232 (private 0.73318) | Δ vs baseline LB: +0.0155. Small local gain, larger LB gain (domain shift) |
| 2026-11-07 | Jamie | Ablation `seclabel`: baseline + secondary labels as soft targets 0.5 (m1-train-seclabel v1, m1-infer-seclabel v1) | 0.9692 (epoch 6) | 0.72916 (private 0.73318) | Δ vs baseline LB: +0.0124. No local gain (val scores primary label only), but helps on LB; still overfits after epoch 6 |
| 2026-11-07 | Jamie | Ablation `focal`: baseline + focal loss γ = 2 instead of BCE (m1-train-focal v1, m1-infer-focal v1) | 0.9733 (epoch 5) | 0.70528 (private 0.70984) | Δ vs baseline LB: −0.0115 (private +0.0108), i.e. no clear effect. Learns faster but peaks at epoch 5 and overfits |
| 2026-11-07 | Jamie | **Baseline v2**: crop + noise/mixup + secondary labels, BCE (m1-train-v2 v1, m1-infer-v2 v1) | 0.9766 (epoch 14) | **0.76299 (private 0.78864)** | Δ vs baseline LB: +0.0462 (private +0.0896). Best private score; 0.014 below top-70% cut-off |
| 2026-11-07 | Jamie | Remove-one `v2-nocrop`: v2 without random crop (m1-train-v2-nocrop v1) | 0.9735 (epoch 12) | 0.75280 (private 0.77176) | Δ vs v2: −0.0102 (private −0.0169) |
| 2026-11-07 | Jamie | Remove-one `v2-nonoise`: v2 without noise + mixup (m1-train-v2-nonoise v1) | 0.9809 (epoch 10) | 0.76740 (private 0.76030) | Δ vs v2: +0.0044 (private −0.0283). Public and private disagree; noise helps on private |
| 2026-11-07 | Jamie | Remove-one `v2-nosec`: v2 without secondary labels (m1-train-v2-nosec v1) | 0.9791 (epoch 15) | 0.73731 (private 0.77243) | Δ vs v2: −0.0257 (private −0.0162). Largest drop |
| 2026-11-07 | Jamie | Add-one `v2-focal`: v2 with focal loss γ = 2 instead of BCE (m1-train-v2-focal v1) | 0.9782 (epoch 15) | 0.74281 (private 0.78241) | Δ vs v2: −0.0202 (private −0.0062). Focal still does not help |
| 2026-11-08 | Sudhishna | SED `sed20`: EfficientNet-B0 + GeM + frame and attention heads, 20 s chunks on Jamie's 128-band mels, v2 recipe, 0.5 attention + 0.5 peak BCE, no overlap at inference (m2-train-sed20 v1, m2-infer-sed20 v1) | 0.9699 first 5 s; 0.9860 peak over 20 s (epoch 7) | 0.72631 (private 0.74369) | Δ vs v2: −0.0367 (private −0.0450). Below v2. Weights saved at epoch 7, when the learning rate was still about 0.64 × peak |
| 2026-11-08 | Sudhishna | SED `sed20-overlap`: same weights as `sed20`, chunks every 5 s instead of every 20 s, predictions averaged (m2-infer-sed20-overlap v1) | same as `sed20` | 0.73212 (private 0.75164) | Δ vs `sed20`: +0.0058 (private +0.0080). Both up but under 0.01, so treated as a tie. Δ vs v2: −0.0309 (private −0.0370) |
| 2026-11-08 | Sudhishna | SED `sed20-last`: same training as `sed20` (history matches to 4 decimals), but saves the final epoch instead of the highest peak score (m2-train-sed20-last v1, m2-infer-sed20-last v1) | 0.9736 first 5 s; 0.9850 peak over 20 s (epoch 15) | 0.73954 (private 0.74880) | Δ vs `sed20`: +0.0132 (private +0.0051). Δ vs v2: −0.0235 (private −0.0398) |
| 2026-11-08 | Sudhishna | SED `sed20-last-overlap`: `sed20-last` weights with chunks every 5 s (m2-infer-sed20-last-overlap v1) | same as `sed20-last` | 0.74830 (private 0.76338) | Δ vs `sed20-last`: +0.0088 (private +0.0146). Δ vs `sed20`: +0.0220 (private +0.0197). Δ vs v2: −0.0147 (private −0.0253) |
| 2026-11-08 | Sudhishna | SED `sed20-hires`: `sed20-last` on spectrograms with the 1st-place write-up's settings (224 bands, n_fft 4096, hop 1252, 0–16 kHz; m2-make-mels-224 v1, m2-train-sed20-hires v1, m2-infer-sed20-hires v1) | 0.9723 first 5 s; 0.9835 peak over 20 s (epoch 15) | 0.72769 (private 0.75484) | Δ vs `sed20-last`: −0.0119 (private +0.0060). Public and private disagree; no gain |
| 2026-11-08 | Sudhishna | SED `sed20-hires-overlap`: `sed20-hires` weights with chunks every 5 s (m2-infer-sed20-hires-overlap v1) | same as `sed20-hires` | 0.73957 (private 0.76079) | Δ vs `sed20-last-overlap`: −0.0087 (private −0.0026). No gain from the write-up's spectrogram settings |
| 2026-11-08 | Sudhishna | SED `sed5`: `sed20-last` on 5 s chunks (10 time steps of 0.5 s), one chunk per 5 s window, so no overlap (m2-train-sed5 v1, m2-infer-sed5 v1) | 0.9798 (epoch 15; first 5 s and peak are the same for 5 s chunks) | **0.77298 (private 0.78740)** | Δ vs v2: +0.0100 (private −0.0012), i.e. level with v2; best public score so far. Δ vs `sed20-last` (also no overlap): +0.0334 (private +0.0386), so the 20 s chunk, not the SED head, held SED back. CPU estimate 16 min |

Leaderboard band cut-offs (public): top 10% ≥ 0.8808, top 30% ≥ 0.8731, top 50% ≥ 0.8102, top 70% ≥ 0.7764. Best: 0.9338.

### Jamie's ablations (baseline improvements)

Each experiment is the baseline with **one** switch turned on, so a change in score comes from
that switch alone. Everything else is identical: EfficientNet-B0, AdamW 1e-3 + one-cycle,
batch 64, 15 epochs, best epoch kept, same 80/20 split (seed 42). Local CV is always scored
on the first 5 s of each hold-out clip with the primary label only, so the numbers are comparable.

**Shared input:** `jmemei/sc4000-birdclef-m1-make-mels`, whole-clip log-mels (≤ 30 s,
stored as uint8 over 120 dB) plus 1,000 unlabelled soundscapes for noise.
Code: `notebooks/member1_train/train.ipynb` (master), copied into the per-experiment folders by
`python3 scripts/m1_experiments.py`.

| Experiment | Switch (vs baseline) | Problem targeted | Training kernel | Inference kernel |
|---|---|---|---|---|
| `base` | none: new data pipeline only; should match baseline CV ≈ 0.973 | (sanity check) | `m1-train-base` | `m1-infer-base` |
| `crop` | random 5 s crop from the whole clip instead of the first 5 s | weak labels | `m1-train-crop` | `m1-infer-crop` |
| `noise` | add soundscape noise (p = 0.5, −20 to −3 dB) + mixup (p = 0.5, Beta(0.4, 0.4), targets = max) | domain shift, overfitting | `m1-train-noise` | `m1-infer-noise` |
| `seclabel` | secondary labels as soft targets (0.5) | label noise | `m1-train-seclabel` | `m1-infer-seclabel` |
| `focal` | focal loss (γ = 2) instead of BCE | class imbalance | `m1-train-focal` | `m1-infer-focal` |
| `v2` | crop + noise/mixup + seclabel (the switches that improved the public LB), BCE | all | `m1-train-v2` | `m1-infer-v2` |
| `v2-nocrop` / `v2-nonoise` / `v2-nosec` | v2 with one switch removed (remove-one ablation) | — | `m1-train-v2-…` | `m1-infer-v2-…` |
| `v2-focal` | v2 with focal loss instead of BCE | — | `m1-train-v2-focal` | `m1-infer-v2-focal` |

Kernel names are short for `jmemei/sc4000-birdclef-<name>`.

**Running one experiment** (e.g. `crop`):

```bash
bash scripts/kaggle_run.sh notebooks/member1_train_crop    # GPU training, ~20 min
bash scripts/kaggle_run.sh notebooks/member1_infer_crop    # CPU inference dry run
kaggle competitions submit birdclef-2025 -k jmemei/sc4000-birdclef-m1-infer-crop -v <version> \
  -f submission.csv -m "M1 crop" --wait
```

The local CV is printed at the end of the training log
(`[crop] best val macro ROC-AUC: ...`) and saved in `outputs/member1_train_crop/cfg.json`
as `best_val_auc`.

**Row template** to append to the experiment log above after each submission:

```
| YYYY-MM-DD | Jamie | Ablation `crop`: baseline + random 5 s crops (kernel m1-train-crop vN) | 0.9xxx (epoch N) | 0.xxxxx (private 0.xxxxx) | Δ vs baseline LB: +0.0xxx |
```

## Team & contributions

| Member | Role |
|--------|------|
|        |      |

## References / acknowledgements

List any public notebooks or code consulted (the course requires acknowledging them in the report).
