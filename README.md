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
| 2026-11-01 | Member 1 | Baseline: EfficientNet-B0 (ImageNet), first 5 s log-mel, primary label only, BCE, no augmentation, 15 epochs (best = epoch 7) | 0.9732 | 0.71681 (private 0.69908) | Rank ~1600/2032 (top 79%). Overfits after epoch 7; large gap between local CV and LB because training clips differ from soundscapes |

Leaderboard band cut-offs (public): top 10% ≥ 0.8808, top 30% ≥ 0.8731, top 50% ≥ 0.8102, top 70% ≥ 0.7764. Best: 0.9338.

### Member 1 ablations (baseline improvements)

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
| `v2` | the switches above that improved the public LB, combined | all | `m1-train-v2` | `m1-infer-v2` |

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
| YYYY-MM-DD | Member 1 | Ablation `crop`: baseline + random 5 s crops (kernel m1-train-crop vN) | 0.9xxx (epoch N) | 0.xxxxx (private 0.xxxxx) | Δ vs baseline LB: +0.0xxx |
```

## Team & contributions

| Member | Role |
|--------|------|
|        |      |

## References / acknowledgements

List any public notebooks or code consulted (the course requires acknowledging them in the report).
