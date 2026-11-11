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

## Team & contributions

| Member | Role |
|--------|------|
|        |      |

## References / acknowledgements

List any public notebooks or code consulted (the course requires acknowledging them in the report).
