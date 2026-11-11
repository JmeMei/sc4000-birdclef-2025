# BirdCLEF+ 2025: pretrained models, adaptation and a final ensemble (NTU SC4000)

## Overview

Late-submission work for the NTU SC4000 team project on Kaggle BirdCLEF+ 2025. The task is to predict 206 classes (birds, amphibians, insects, mammals) for every 5 s window of about 700 one-minute test soundscapes; the metric is macro ROC-AUC, and the submission notebook must run on CPU in at most 90 min with internet off. Every experiment follows a "v2 + one change" protocol: our team's baseline v2 recipe, split (seed 42, stratified 80/20) and inference protocol stay fixed, and only one component changes.

We compare three pretraining paradigms: a bird-pretrained CNN backbone (A1), Perch 2.0 frozen embeddings (A2) and Bird-MAE-Base frozen features (A3). Label-free statistics-level alignment (AdaBN, CORAL, mean/std matching) hurt every model it was applied to, whereas content-level adaptation (distillation and noisy student on unlabelled soundscapes) helped. The final ensemble E4 scores 0.88749 public / 0.88029 private, against 0.763 / 0.789 for v2.

## Scope of this repository

This repository covers our pretrained-model and inference-speed work: pretrained models A1, A2 and A3, CPU inference speed (profiling and a faster pipeline), the adaptation study (AdaBN, CORAL), distillation and noisy student round 1. It also contains the final ensemble, E1 to E4.

Everything builds on our team's baseline model (v2): [`sc4000-birdclef-m1-make-mels`](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-make-mels) (spectrogram cache and split) and [`sc4000-birdclef-m1-train-v2`](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-train-v2) (trained EfficientNet-B0), plus the inference notebook [`sc4000-birdclef-m1-infer-v2`](https://www.kaggle.com/code/jmemei/sc4000-birdclef-m1-infer-v2). Local copies are in `m1/`.

## Status (as of 10 Nov 2026)

**Experiments done.** We made 19 late submissions, all scored, from 40 Kaggle notebooks, all COMPLETE and all private.

**Deliverables:**

- `docs/NOTEBOOKS.md`: catalogue of the 40 notebooks, 2 private datasets and the upstream inputs.
- `docs/DOCUMENTATION.md`: technical documentation (17 sections; section 16 lists all submissions, section 17 the compute and limitations).
- `docs/paper/`: AAAI-27 paper (`main.tex`, `references.bib`, `paper.docx`, `figures/`), plus `docs/paper_overleaf.zip` for Overleaf.
- `reports/`: `results_table.md`, `timing_table.md`, `ensemble_handoff.md`, `pretrained_models_section.md`.

**Not done.** We did not export the CNN to ONNX/OpenVINO (the submitted CNNs run with torch on CPU; E4 takes 32.3 min measured). The paper has not been compiled locally (no LaTeX on the laptop), so it should be compiled on Overleaf. Sharing the private Kaggle notebooks and datasets with teammates has to be done by hand in the Kaggle UI (steps in `reports/ensemble_handoff.md`).

## Results summary

Reference: v2 public 0.763 / private 0.789; grading cut-off 0.776 public; team goal 0.810. CPU minutes are for 700 one-minute soundscapes; "measured" is a full 700-file run, "projected" is scaled up from a 20-file dry run.

| System | Ref | Public | Private | Δ public vs v2 | CPU min / 700 files |
|---|---|---|---|---|---|
| v2 baseline (team baseline, EfficientNet-B0) | n/a | 0.763 | 0.789 | 0 | 11.1 measured (5.5 with the fast pipeline) |
| A1 bird-pretrained EfficientNetV2-S | 58120958 | 0.82822 | 0.81642 | +0.0652 | about 37 projected (8.0 measured, fast pipeline) |
| A2 Perch 2.0 + MLP head | 58122219 | 0.84774 | 0.85226 | +0.0847 | 25.0 projected |
| A3 Bird-MAE-Base + MLP head | 58127656 | 0.77640 | 0.78016 | +0.0134 | 61.6 projected |
| A3 + Perch-distilled | 58161782 | 0.81238 | 0.80208 | +0.0494 | 67.5 projected |
| A2 + A1-distilled (A2d, best single model) | 58164973 | 0.86651 | 0.85737 | +0.1035 | 26.5 projected |
| Noisy student MAIN | 58181262 | 0.85985 | 0.85185 | +0.0968 | 9.9 measured |
| Noisy student ABLATION | 58181267 | 0.84003 | 0.83794 | +0.0770 | 10.0 measured |
| E1 MAIN + A2d, rank average | 58196484 | 0.88352 | 0.87203 | +0.1205 | 28.3 measured |
| E2 = E1 + temporal smoothing | 58196486 | 0.88662 | 0.88008 | +0.1236 | 30.4 measured |
| E3 = E2 + ABLATION, 1/3 each | 58198886 | 0.88058 | 0.87522 | +0.1176 | 39.9 measured |
| **E4 = MAIN 0.4 + A2d 0.6, smoothed (selected by public LB; E2 is within noise)** | 58198889 | **0.88749** | **0.88029** | **+0.1245** | **32.3 measured** |

All 19 submissions (including the AdaBN and CORAL variants, which all scored below their unadapted parents) are in `docs/DOCUMENTATION.md`, section 16. Public differences below 0.01 are treated as noise.

## Folder structure

```
README.md
datasets/            metadata and small files for the private Kaggle datasets (large files are on Kaggle)
  perch-onnx/  birdmae-base/  ns-pseudo-labels/
docs/
  NOTEBOOKS.md             catalogue of the 40 Kaggle notebooks, datasets, upstream inputs
  DOCUMENTATION.md         technical documentation, sections 1-17
  IMPLEMENTATION_GUIDE.md  earlier guide (8 Nov), superseded by DOCUMENTATION.md
  LOG.md                   chronological run log
  paper/                   AAAI-27 paper: main.tex, references.bib, paper.docx, figures/, UNVERIFIED_REFS.md
  paper_overleaf.zip       paper upload for Overleaf
  paper_v2/, paper_v2_overleaf.zip   earlier Bird-MAE-centred paper version
  presentation/, presentation_v2/    Beamer slides and speaker scripts (earlier versions)
  _kaggle_logs/            downloaded run logs and CLI output used as evidence
m1/                  our team's baseline v2 notebooks
notebooks/           40 folders, one per Kaggle notebook (notebook + kernel-metadata.json)
notes/               research notes (papers, model formats, licences, CLI, citations)
reports/             results_table.md, timing_table.md, ensemble_handoff.md, pretrained_models_section.md
```

## How to reproduce

Requirements: a Kaggle account that has accepted the BirdCLEF+ 2025 competition rules, and read access to the private items under `strippedbanana` (all notebooks and datasets are private). Each notebook's inputs are listed in `docs/NOTEBOOKS.md` and in its `kernel-metadata.json` under `notebooks/<folder>/`. GPU notebooks need a T4 and print or raise a GPU guard if none is available. All inference notebooks are CPU-only with internet off.

Private datasets to attach (both under `strippedbanana`):

- [`sc4000-birdclef-m3-perch-onnx`](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-perch-onnx): community ONNX conversion of Perch 2.0.
- [`sc4000-birdclef-m3-birdmae-base`](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-birdmae-base): Bird-MAE-Base weights (not needed for E4).

Large files are not stored in this repository. Get them from these private Kaggle datasets:

- `datasets/perch-onnx/perch_v2.onnx` (390 MB): [`sc4000-birdclef-m3-perch-onnx`](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-perch-onnx).
- `datasets/birdmae-base/model.safetensors` (326 MB): [`sc4000-birdclef-m3-birdmae-base`](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-birdmae-base).
- `datasets/ns-pseudo-labels/pseudo_main.npy` and `pseudo_ablation.npy` (92 MB each): [`sc4000-birdclef-m3-ns-pseudo-labels`](https://www.kaggle.com/datasets/strippedbanana/sc4000-birdclef-m3-ns-pseudo-labels).

Upstream: our team's baseline notebooks `make-mels` and `train-v2` notebooks, and the public dataset [`vladimirsydor/bird-clef-2025-all-pretrained-models`](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-all-pretrained-models).

Run order by phase (notebook names omit the prefix `sc4000-birdclef-m3-`; the accelerator is in brackets):

| Phase | Notebooks in order |
|---|---|
| 1. Pretrained models and speed | [`a1-v2s-train`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-v2s-train) (T4); [`a1-v2s-infer`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-v2s-infer) (CPU); [`a2-perch-embed`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-embed) (T4); [`a2-perch-infer`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-perch-infer) (CPU); [`a3-birdmae-embed`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-birdmae-embed) (T4); [`a3-probe-retry`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-probe-retry) (CPU); [`a3-mlp-head`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-mlp-head) (CPU); [`a3-birdmae-infer`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-birdmae-infer) (CPU); [`profile-infer-v2`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-profile-infer-v2) (CPU); [`fast-infer`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-fast-infer) (CPU) |
| 2. Adaptation (AdaBN, CORAL) | [`a1-adabn`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-adabn) (T4); [`a1-adabn-fast-infer`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a1-adabn-fast-infer) (CPU); [`v2-adabn`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-v2-adabn) (T4); [`v2-adabn-fast-infer`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-v2-adabn-fast-infer) (CPU); [`soundscape-embed`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-soundscape-embed) (T4); [`coral-heads`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-coral-heads) (CPU); [`a2-coral-std-infer`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-coral-std-infer) (CPU); [`a2-meanstd-infer`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-meanstd-infer) (CPU); [`a3-coral-std-infer`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-coral-std-infer) (CPU); [`a3-meanstd-infer`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-meanstd-infer) (CPU); [`coral-diag`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-coral-diag) (CPU); [`head-seed-repeats`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-head-seed-repeats) (CPU) |
| 3. Distillation | [`a3-distill`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-distill) (CPU); [`a3-distill-infer`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-distill-infer) (CPU); [`a3-distill-coral-infer`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a3-distill-coral-infer) (CPU); [`a2-a1distill`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-a1distill) (T4); [`a2-a1distill-infer`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-a2-a1distill-infer) (CPU) |
| 4. Noisy student | [`ns-perch-all`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-perch-all) (T4); [`ns-a1-all`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-a1-all) (T4); [`ns-pseudo-labels`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-pseudo-labels) (CPU); [`ns-mel-check`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-mel-check) (T4); [`ns-student-main`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-main) (T4); [`ns-student-ablation`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-ablation) (T4); [`ns-student-main-infer`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-main-infer) (CPU); [`ns-student-ablation-infer`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-student-ablation-infer) (CPU); [`ns-head-seeds`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ns-head-seeds) (CPU) |
| 5. Final ensemble | [`ens-e1`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ens-e1) (CPU); [`ens-e2`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ens-e2) (CPU); [`ens-e3`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ens-e3) (CPU); [`ens-e4`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ens-e4) (CPU) |

Approximate GPU use (all on a T4, from `docs/DOCUMENTATION.md` section 17): about 5.6 h in total against a 30 h weekly quota. The largest runs are `ns-student-main` (1.18 h), `ns-student-ablation` (1.10 h), `ns-perch-all` (0.91 h) and `a2-perch-embed` (about 0.9 h); `a1-v2s-train` took about 0.5 h and `a3-birdmae-embed` about 0.3 h. Everything else ran on Kaggle CPU.

### Shortest path to the final submission (E4)

Retraining is not needed: the weights that scored are the kernel outputs listed below.

1. Open [`ens-e4`](https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ens-e4), choose Copy & Edit, and attach the inputs listed in `reports/ensemble_handoff.md`:
   - competition data `birdclef-2025`;
   - kernel output `ns-student-main` (CNN weights);
   - kernel output `a2-a1distill` (Perch head);
   - dataset `sc4000-birdclef-m3-perch-onnx`;
   - kernel output `a2-perch-embed` (ONNX Runtime CPU wheels, since internet is off);
   - kernel output `ns-student-ablation` is also attached to the original notebook because E3 needs it; E4 does not load it, so it can be removed from a copy.
2. Settings: CPU (no accelerator), internet off. Save Version, then Save & Run All. On the public (empty) test folder it runs as a dry run on 700 `train_soundscapes` and prints the projected time and a sanity-check result; on the hidden test set it produces the real `submission.csv`.
3. Runtime is about 32.3 min for 700 files (measured), within the 75 min budget and the 90 min limit.
4. Submit from the competition page (Late Submission, Notebook). `ens-e2` is an equally good fallback.

The four ensemble notebooks differ only in the line `VARIANT = "E..."`.

## Known issues / limitations

- Single validation split (seed 42), inherited from v2; local validation did not predict the leaderboard.
- Seeds were repeated for the small heads only; the CNNs (A1 and the students) were trained once.
- Public and private scores disagree on close calls; differences below 0.01 are treated as noise (E4 vs E2 is +0.0009 public / +0.0002 private).
- Perch 2.0 was released on 6 Aug 2025, after BirdCLEF 2025 closed on 5 Jun 2025, so A2 and everything built on it are late-submission results. The ONNX file is a community conversion. Perch was trained on Xeno-canto and iNaturalist, which `train_audio` also draws from, so A2's validation AUC may be inflated (overlap not checked).
- Bird-MAE declares no licence for its weights, so its dataset stays private; Bird-MAE is not part of the final ensemble.
- The MAIN teacher was chosen after the A2d score was known.
- MAIN and ABLATION differ in both teacher and label density (2.50% vs 0.74% non-zero), so their difference cannot be attributed to the teacher alone.
- The kernel version of the A1 control (ref 58159079, `fast-infer`) was not recorded.
- The v2 runtime of about 25 min reported for our team's baseline was not reproduced (11.1 min measured here); the cause is unknown.
- "Projected" CPU times scale a 20-file dry run and may be pessimistic; the scored runtime is not available from the Kaggle CLI.
- A1 is a backbone change from B0 to EfficientNetV2-S, not a weights-only swap.
- ONNX Runtime / OpenVINO export of the CNN was not done.
- Some references could be only partly verified; see `docs/paper/UNVERIFIED_REFS.md`.

## Other docs

- [`docs/NOTEBOOKS.md`](docs/NOTEBOOKS.md): the 40 notebooks, with links, inputs, results and paste-ready descriptions.
- [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md): full technical documentation.
- [`docs/paper/`](docs/paper/): AAAI-27 paper source, `paper.docx`, figures and [`UNVERIFIED_REFS.md`](docs/paper/UNVERIFIED_REFS.md); [`docs/paper_overleaf.zip`](docs/paper_overleaf.zip) for Overleaf.
- [`docs/IMPLEMENTATION_GUIDE.md`](docs/IMPLEMENTATION_GUIDE.md): the earlier 8 Nov guide, superseded by `DOCUMENTATION.md`.
- [`docs/LOG.md`](docs/LOG.md): chronological log.
- [`reports/results_table.md`](reports/results_table.md), [`reports/timing_table.md`](reports/timing_table.md), [`reports/ensemble_handoff.md`](reports/ensemble_handoff.md), [`reports/pretrained_models_section.md`](reports/pretrained_models_section.md).
- Earlier versions: `docs/paper_v2/` (a Bird-MAE-centred reframing of the same results), `docs/presentation/`, `docs/presentation_v2/`.
