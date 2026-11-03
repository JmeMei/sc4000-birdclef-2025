# Remaining Work and Task Split

SC4000 BirdCLEF+ 2025 group project, 4 members. Written 3 Nov 2026.

**Deadline: Fri 13 Nov 2026, 11:59pm.** Hand in on NTULearn: report (AAAI, max 10 pages), video link (.txt), and the submission file.

Rename "Member 3" and "Member 4" to your real names.

## Already done

| Done | By |
|---|---|
| Private GitHub repo and VS Code → Kaggle workflow (`scripts/kaggle_run.sh`) | Jamie |
| EDA notebook and 7 report figures (`notebooks/eda`, `report/figures`) | Jamie |
| Baseline model: EfficientNet-B0 on first-5 s mel spectrograms (local val AUC 0.9732) | Jamie |
| CPU/offline inference notebook (about 25 min, under the 90 min limit) | Jamie |
| First submission: **public 0.71681**, rank about 1600 / 2032 (top 79%) | Jamie |
| Draft report: introduction and data section (`report/progress_report.md`) | Jamie |

## Team targets

| | Public LB score | Grade band |
|---|---|---|
| Now | 0.717 | bottom 30% |
| **Minimum target** | **≥ 0.776** | top 70% |
| **Main target** | **≥ 0.810** | top 50% |
| Stretch | ≥ 0.873 | top 30% |

These are targets, not guarantees. Every experiment must be submitted and logged, even if it scores lower, because the ablation results are part of the grade.

## The split

Each person owns **one modelling track** and **one share of the report and video**, so the work is roughly equal.

### Jamie: improve the baseline (ablations)
Branch: `jamie/baseline-ablations`. Notebooks go in `notebooks/jamie_*`.

1. Save the mel spectrograms as a private Kaggle dataset, so every training run is about twice as fast. Share it with the team.
2. Ablation: random 5 s crops from the whole clip instead of the first 5 s.
3. Ablation: background noise from the unlabelled soundscapes, plus mixup.
4. Ablation: use secondary labels and Focal loss.
5. Combine the ideas that helped into "baseline v2" and submit it.
6. Admin: add teammates as GitHub collaborators; screenshot the baseline score.
7. Report: **Experiments and results** section, with the ablation table and discussion.

**Expected results**
- A Kaggle dataset of saved spectrograms that the whole team uses.
- 4 or more leaderboard submissions, each one logged in the README.
- An ablation table: baseline vs. each change on its own vs. v2.
- A v2 model scoring **≥ 0.776** public (target).
- The Experiments section, about 1.5 pages.

### Arsh: SED model
Branch: `arsh/sed-model`. Notebooks go in `notebooks/arsh_*`.

1. Build a sound event detection (SED) model on 20 s chunks. It predicts when a call happens inside the clip.
2. Re-implement it from the 1st-place *write-up* only. Never copy their code. Cite it in the report.
3. Write a CPU inference notebook for it that finishes in under 90 min, and submit it.
4. Hand trained model weights to Member 4 for the ensemble by **8 Nov**.
5. Report: **Methodology** section (preprocessing, CNN baseline, SED model), with the reason for each choice.
6. Video: make the slide deck that everyone presents from.

**Expected results**
- A working SED training and inference notebook pair.
- A leaderboard score, compared against the baseline (target: **beat baseline v2**).
- A diagram or short description of the SED architecture for the report.
- The Methodology section, about 1.5 pages, and the slide deck.

### Member 3: pretrained audio model and inference speed
Branch: `member3/birdmae-speed`. Notebooks go in `notebooks/member3_*`.

1. Run a Bird-MAE pilot on 1 fold. Check that it fits the 90 min CPU limit before scaling up.
2. Measure CPU inference time for every model (baseline, SED, Bird-MAE). This tells the team how many models fit in the final ensemble.
3. Speed up inference (for example, an ONNX or OpenVINO export) so that more models fit in 90 min.
4. Report: **Introduction** (competition, challenges, related work) and the **References** list.
5. Video: edit the recorded clips together, upload to YouTube as private, and write the link .txt file.

**Expected results**
- A go/no-go decision on Bird-MAE, with its leaderboard score.
- A timing table: model vs. CPU minutes for the full test set.
- A faster inference notebook (target: **at least 2× faster** than now).
- The Introduction and References, and the finished video uploaded.

### Member 4: pseudo-labelling, ensemble and final submission
Branch: `member4/pseudolabel-ensemble`. Notebooks go in `notebooks/member4_*`.

1. Build the pseudo-labelling pipeline: predict on the 9,726 unlabelled soundscapes. Start with the baseline model as the teacher now, and swap in the best model later.
2. Retrain using the confident pseudo-labels (1 round) and submit.
3. Build the ensemble notebook that blends the best models from everyone, within 90 min on CPU.
4. Lock in the final submission by **10 Nov**. Take leaderboard screenshots and compute the relative ranking.
5. Report: **Abstract**, **Conclusion**, **team contributions**, and the final **AAAI format check** (max 10 pages incl. references; breaking it costs a 50% penalty).
6. Hand in on NTULearn on 13 Nov.

**Expected results**
- Pseudo-labelled data, and a model retrained on it, with its leaderboard score.
- The final ensemble notebook and its submission (team target **≥ 0.810**).
- Screenshots and the relative-ranking calculation for the report.
- Abstract, Conclusion and contributions written; the report formatted and submitted.

### Everyone
- Record your own part of the video (max 10 min total; every member must present).
- Add every experiment to the **Experiment log** in `README.md`.
- Write 2–3 sentences about your own contribution for the report.

## Timeline

| Date | Milestone |
|---|---|
| 3–4 Nov | Create branches. Jamie shares the saved-spectrogram dataset. |
| 5–7 Nov | Run the modelling tracks in parallel and submit each experiment. |
| **8 Nov** | **Merge all model branches into `main`.** Hand weights to Member 4. |
| 9–10 Nov | Pseudo-labelling round and ensemble. **Final submission locked on 10 Nov.** |
| 10–11 Nov | Report sections merged into `main`. Assemble the full draft in the AAAI template. |
| 12 Nov | Record and upload the video. Final read-through and format check. |
| **13 Nov** | **Submit on NTULearn before 11:59pm.** |

## Git workflow

Each person works on their own branch and merges into `main` through a pull request when finished.

**Branches**

| Person | Branch |
|---|---|
| Jamie | `jamie/baseline-ablations` |
| Arsh | `arsh/sed-model` |
| Member 3 | `member3/birdmae-speed` |
| Member 4 | `member4/pseudolabel-ensemble` |

**First time: create your branch**

```bash
git checkout main
git pull
git checkout -b arsh/sed-model
git push -u origin arsh/sed-model
```

**Each day: save your work**

```bash
git add -A
git commit -m "Short description of what changed"
git push
```

**When you're done: merge into `main`**
1. Get the latest `main` into your branch first: `git checkout main && git pull && git checkout <your-branch> && git merge main`.
2. On GitHub, open a **Pull request** from your branch into `main`.
3. Ask one teammate to look it over, then click **Merge**.

**Avoiding merge conflicts**
- Only edit files in your own `notebooks/<name>_*` folders.
- Write your report sections as separate files in `report/sections/`: `01-introduction.md`, `02-methodology.md`, `03-experiments.md`, `04-conclusion.md`. Member 4 assembles them into the AAAI template at the end.
- In the README experiment log, only **add rows at the bottom**. Never edit other people's rows.
- Never commit data, model weights or `kaggle.json`; the `.gitignore` already blocks them.

## Rules (from the course brief)
- **Never copy other Kaggle participants' code.** That counts as plagiarism. Ideas from write-ups are fine if cited.
- Keep the repo and every Kaggle notebook **private**.
- Each person trains on their **own Kaggle account**. That's 30 GPU hours a week each; check with `kaggle quota`.
