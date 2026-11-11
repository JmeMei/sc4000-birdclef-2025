# Remaining Work and Task Split

SC4000 BirdCLEF+ 2025 group project, 4 members. Updated 3 Nov 2026.

**Deadline: Fri 13 Nov 2026, 11:59pm.** Hand in on NTULearn: report (AAAI, max 10 pages), video link (.txt), and the submission file.

## Already done (Jamie)

| Done |
|---|
| Private GitHub repo and VS Code → Kaggle workflow (`scripts/kaggle_run.sh`) |
| EDA notebook and 7 report figures (`notebooks/jamie_eda`, `report/figures`) |
| Baseline model: EfficientNet-B0 on first-5 s mel spectrograms (local val AUC 0.9732) |
| CPU/offline inference notebook (about 25 min, under the 90 min limit) |
| First submission: **public 0.71681**, rank about 1600 / 2032 (top 79%) |
| Draft report: introduction and data section (`report/progress_report.md`) |

Every task listed under each member below is **still to do**.

## Team targets

| | Public LB score | Grade band |
|---|---|---|
| Now | 0.717 | bottom 30% |
| **Minimum target** | **≥ 0.776** | top 70% |
| **Main target** | **≥ 0.810** | top 50% |
| Stretch | ≥ 0.873 | top 30% |

These are targets, not guarantees. Submit and log every experiment, even ones that score lower: the ablation results are part of the grade.

## Can we work at the same time?

**Yes. The four tracks run in parallel**, so nobody needs to wait for anyone to start. There are only three hand-offs:

| Hand-off | From → to | By |
|---|---|---|
| Saved-spectrogram Kaggle dataset (optional speed-up; others can start without it) | Jamie → everyone | 5 Nov |
| Trained model weights for the ensemble | Jamie, Sudhishna, Arshiya → Arshiya | 8 Nov |
| Report sections for assembly into the AAAI template | everyone → Arshiya (Member 4 role) | 11 Nov |

Arshiya (Member 4 role) can start pseudo-labelling now using the existing baseline model as the teacher, then swap in a better model after 8 Nov.

Only the end is sequential: final ensemble → report results → video → hand-in.

## Time estimates

The times below are hands-on work. Kaggle runs add waiting time on top: about 30 min per training run and about 40 min for each submission to be scored. Plan to start runs and work on writing while they finish.

| Member | Hands-on hours |
|---|---|
| Jamie | about 15 h (lighter, since Jamie already did the setup and baseline) |
| Sudhishna | about 18 h |
| Arshiya (Member 3 role) | about 18 h |
| Arshiya (Member 4 role) | about 19 h |

## The split

Each person owns **one modelling track** and **one share of the report and video**.

### Jamie: improve the baseline (ablations)
Branch: `member1/baseline-ablations`. Notebooks go in `notebooks/jamie_*`.

| # | Task | Time |
|---|---|---|
| 1 | Add teammates as GitHub collaborators (Settings → Collaborators) | 10 min |
| 2 | Screenshot the baseline score (Kaggle → Submissions tab) and its leaderboard position | 10 min |
| 3 | Save mel spectrograms of the **whole** clip (not just the first 5 s) as a private Kaggle dataset and share it with the team | 1.5 h |
| 4 | Ablation: random 5 s crops from the whole clip instead of the first 5 s | 2 h |
| 5 | Ablation: background noise from the unlabelled soundscapes, plus mixup | 3 h |
| 6 | Ablation: secondary labels and Focal loss | 2 h |
| 7 | Combine the ideas that helped into "baseline v2" and submit it | 2 h |
| 8 | Report: **Experiments and results** section, with the ablation table and discussion | 4 h |

**Expected results**
- A shared Kaggle dataset of spectrograms.
- 4 or more leaderboard submissions, each logged in the README.
- An ablation table: baseline vs. each change on its own vs. v2.
- A v2 model scoring **≥ 0.776** public (target).
- The Experiments section, about 1.5 pages.

### Sudhishna: SED model
Branch: `member2/sed-model`. Notebooks go in `notebooks/sudhishna_*`.

| # | Task | Time |
|---|---|---|
| 1 | Build a sound event detection (SED) model on 20 s chunks; it predicts when a call happens inside the clip. Re-implement it from the 1st-place *write-up* only, never their code, and cite it. | 6 h |
| 2 | Train it on Kaggle and tune it | 2 h |
| 3 | Write a CPU inference notebook that finishes in under 90 min, and submit it | 3 h |
| 4 | Hand the trained model weights to Arshiya (Member 4 role) by **8 Nov** | 15 min |
| 5 | Report: **Methodology** section (preprocessing, CNN baseline, SED model), with the reason for each choice | 4 h |
| 6 | Video: make the slide deck everyone presents from | 3 h |

**Expected results**
- A working SED training and inference notebook pair.
- A leaderboard score compared against the baseline (target: **beat baseline v2**).
- A diagram or short description of the SED architecture.
- The Methodology section (about 1.5 pages) and the slide deck.

### Arshiya (Member 3 role): pretrained audio model and inference speed
Branch: `member3/birdmae-speed`. Notebooks go in `notebooks/arshiya_*`.

| # | Task | Time |
|---|---|---|
| 1 | Bird-MAE pilot on 1 fold; check it fits the 90 min CPU limit before scaling up | 6 h |
| 2 | Measure CPU inference time for every model (baseline, SED, Bird-MAE), so the team knows how many fit in the ensemble | 2 h |
| 3 | Speed up inference (for example an ONNX or OpenVINO export) so more models fit in 90 min | 4 h |
| 4 | Report: **Introduction** (competition, challenges, related work) and the **References** list | 3 h |
| 5 | Video: edit the recorded clips together, upload to YouTube as private, and write the link .txt file | 3 h |

**Expected results**
- A go/no-go decision on Bird-MAE, with its leaderboard score.
- A timing table: model vs. CPU minutes for the full test set.
- A faster inference notebook (target: **at least 2× faster**).
- The Introduction and References, and the finished video uploaded.

### Arshiya (Member 4 role): pseudo-labelling, ensemble and final submission
Branch: `member4/pseudolabel-ensemble`. Notebooks go in `notebooks/arshiya_*`.

| # | Task | Time |
|---|---|---|
| 1 | Pseudo-labelling pipeline: predict on the 9,726 unlabelled soundscapes, starting with the baseline model as teacher | 5 h |
| 2 | Retrain using the confident pseudo-labels (1 round) and submit | 3 h |
| 3 | Ensemble notebook that blends the best models from everyone, within 90 min on CPU | 4 h |
| 4 | Lock in the final submission by **10 Nov**; take leaderboard screenshots and compute the relative ranking | 1.5 h |
| 5 | Report: **Abstract**, **Conclusion**, **team contributions**, and the final **AAAI format check** (max 10 pages incl. references; breaking it costs a 50% penalty) | 5 h |
| 6 | Hand in on NTULearn on 13 Nov | 30 min |

**Expected results**
- Pseudo-labelled data and a model retrained on it, with its leaderboard score.
- The final ensemble notebook and submission (team target **≥ 0.810**).
- Screenshots and the relative-ranking calculation.
- Abstract, Conclusion and contributions written; the report formatted and submitted.

### Everyone
- Record your own part of the video (max 10 min total; every member presents).
- Add every experiment to the **Experiment log** in `README.md`.
- Write 2–3 sentences about your own contribution for the report.

## Timeline

| Date | Milestone |
|---|---|
| 3–4 Nov | Switch to your branch and start. Jamie adds collaborators. |
| 5 Nov | Jamie shares the saved-spectrogram dataset. |
| 5–7 Nov | Modelling tracks run in parallel; submit each experiment. |
| **8 Nov** | **Merge all model branches into `main`.** Hand weights to Arshiya (Member 4 role). |
| 9–10 Nov | Pseudo-labelling round and ensemble. **Final submission locked on 10 Nov.** |
| 10–11 Nov | Report sections merged into `main`; Arshiya (Member 4 role) assembles the AAAI draft. |
| 12 Nov | Record and upload the video. Final read-through and format check. |
| **13 Nov** | **Submit on NTULearn before 11:59pm.** |

## Git workflow

Each person works on their own branch and merges into `main` through a pull request when finished.

| Person | Branch |
|---|---|
| Jamie | `member1/baseline-ablations` |
| Sudhishna | `member2/sed-model` |
| Arshiya (Member 3 role) | `member3/birdmae-speed` |
| Arshiya (Member 4 role) | `member4/pseudolabel-ensemble` |

**First time: get the repo and switch to your branch**

```bash
git clone https://github.com/JmeMei/sc4000-birdclef-2025.git
cd sc4000-birdclef-2025
git checkout member2/sed-model
```

**Each day: save your work**

```bash
git add -A
git commit -m "Short description of what changed"
git push
```

**When you're done: merge into `main`**
1. Bring the latest `main` into your branch: `git checkout main && git pull && git checkout <your-branch> && git merge main`.
2. On GitHub, open a **Pull request** from your branch into `main`.
3. Ask one teammate to look it over, then click **Merge**.

**Avoiding merge conflicts**
- Only edit files in your own `notebooks/member<N>_*` folders.
- Write report sections as separate files in `report/sections/`: `01-introduction.md`, `02-methodology.md`, `03-experiments.md`, `04-conclusion.md`. Arshiya (Member 4 role) assembles them into the AAAI template.
- In the README experiment log, only **add rows at the bottom**.
- Never commit data, model weights or `kaggle.json`; the `.gitignore` already blocks them.

## Rules (from the course brief)
- **Never copy other Kaggle participants' code.** That counts as plagiarism. Ideas from write-ups are fine if cited.
- Keep the repo and every Kaggle notebook **private**.
- Each person trains on their **own Kaggle account**. That's 30 GPU hours a week each; check with `kaggle quota`.
