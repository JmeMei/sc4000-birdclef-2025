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

## Experiment log

Record every leaderboard submission here — the report needs baselines and ablations.

| Date | Who | Experiment | Local CV AUC | Public LB | Notes |
|------|-----|------------|--------------|-----------|-------|
|      |     |            |              |           |       |

## Team & contributions

| Member | Role |
|--------|------|
|        |      |

## References / acknowledgements

List any public notebooks or code consulted (the course requires acknowledging them in the report).
