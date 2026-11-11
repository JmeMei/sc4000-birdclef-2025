# Hand-off: final ensemble E4

## Final submission

**E4: per-class rank average of the MAIN noisy-student CNN (weight 0.4) and the A2 + A1-distilled Perch head (weight 0.6), with temporal smoothing.**

| | Value |
|---|---|
| Kaggle notebook | `strippedbanana/sc4000-birdclef-m3-ens-e4`, version 1: https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ens-e4 |
| Submission ref | 58198889 |
| Public / private | **0.88749 / 0.88029** |
| CPU time | 32.3 min projected for 700 files, from a 700-file dry run on Kaggle CPU (budget 75, limit 90) |
| Local copy | `notebooks/arshiya_ens_e4/` (the notebook and `kernel-metadata.json`) |

For reference: v2 baseline 0.763 / 0.789; grading cut-off 0.776 public; team goal 0.810; best single model (A2 + A1-distilled) 0.86651 / 0.85737.

## Selection rule

This rule was pre-registered in `STATE.md` at 10 Nov 09:55 SGT, before any ensemble result existed.

The phase had a cap of 6 submissions. We pick by **public** LB and report private too. As a gate, a dry run must project ≤ 75 CPU min on 700 `train_soundscapes`. Sanity checks before submitting:
- columns match `sample_submission`;
- row count = 12 × files, with unique row ids;
- no NaN or inf, and all values in [0, 1];
- per-class Spearman between the ensemble and each single model ≥ 0.5.

E4 has the highest public score. Its lead over E2 (+0.0009 public / +0.0002 private) is far below the 0.01 noise level, so **E2 is an equally good fallback**.

## E1–E4

All variants share one audio decode per file and run on CPU with internet off. Each model's per-class probabilities are ranked over all output rows, the ranks are scaled to [0, 1], and the result is a weighted mean of the ranks.

Smoothing, when on, is applied to each model's probabilities within each file before ranking:
- `p'_t = 0.25 p_{t-1} + 0.5 p_t + 0.25 p_{t+1}`;
- at the first and last window, `(0.5 p_t + 0.25 p_neighbour) / 0.75`.

| Variant | Models (weights) | Smoothing | Public | Private | Δ vs A2d (public / private) | CPU min / 700 | Ref |
|---|---|---|---|---|---|---|---|
| E1 | MAIN 0.5 + A2d 0.5 | no | 0.88352 | 0.87203 | +0.0170 / +0.0147 | 28.3 | 58196484 |
| E2 | MAIN 0.5 + A2d 0.5 | yes | 0.88662 | 0.88008 | +0.0201 / +0.0227 | 30.4 | 58196486 |
| E3 | MAIN 1/3 + ABLATION 1/3 + A2d 1/3 | yes | 0.88058 | 0.87522 | +0.0141 / +0.0179 | 39.9 | 58198886 |
| **E4** | **MAIN 0.4 + A2d 0.6** | **yes** | **0.88749** | **0.88029** | **+0.0210 / +0.0229** | **32.3** | **58198889** |

Kernels: `strippedbanana/sc4000-birdclef-m3-ens-e1` … `-ens-e4`, all version 1. The four notebooks are identical except for the line `VARIANT = "E…"`.

## Inputs E4 needs (attach these)

| Input | Kaggle source | File used | sha256 |
|---|---|---|---|
| Competition data | `birdclef-2025` | `test_soundscapes/`, `sample_submission.csv` | – |
| CNN weights (MAIN noisy student) | kernel output `strippedbanana/sc4000-birdclef-m3-ns-student-main` v1 | `model.pth` (79 MB), `cfg.json`, `species.json` | `d9f74e6f740da79e26e6dc37337e22f1423c4e521cbda1aea00b829f54f99eb8` (model.pth) |
| Perch head (A2 + A1-distilled) | kernel output `strippedbanana/sc4000-birdclef-m3-a2-a1distill` v1 | `perch_head_a1distill.npz` (3.5 MB) | `3200b0a718fa1231b72a378b8e9e977a636f6af9898939b7af5ce6eede671f6e` |
| Perch 2.0 ONNX | dataset `strippedbanana/sc4000-birdclef-m3-perch-onnx` | `perch_v2.onnx` | `bf0c8467a924cb074663970ca4a0ab1e143602121930209657d0dff5d5cefa1f` (checked by the notebook) |
| ONNX Runtime CPU wheels (internet is off) | kernel output `strippedbanana/sc4000-birdclef-m3-a2-perch-embed` v1 | `ort_cpu_wheels/` | – |
| ABLATION student (E3 only; attached but not loaded by E4) | kernel output `strippedbanana/sc4000-birdclef-m3-ns-student-ablation` v1 | `model.pth` | – |

The notebook finds each input by searching `/kaggle/input` for a marker file, so different mount paths are fine. If something is missing, it stops with an error that names the source to attach.

## How to rerun

1. Open https://www.kaggle.com/code/strippedbanana/sc4000-birdclef-m3-ens-e4 and click **Copy & Edit**. This needs read access to all the inputs above; see "Sharing" below.
2. Check that the five sources are attached. Settings: CPU (no accelerator), internet **off**.
3. **Save Version → Save & Run All.**
   - With the public (empty) `test_soundscapes/`, the run is a dry run on 700 `train_soundscapes`. It prints `TIMING | projected 700 files: … min` and `SANITY | PASS`. The `submission.csv` it writes then covers training audio and is not a valid entry.
   - A scored run uses the hidden test set automatically.
4. Submit from the competition page (Late Submission → Notebook → this notebook, output `submission.csv`), or with the CLI: `kaggle competitions submit birdclef-2025 -k <owner>/<kernel> -v <version> -f submission.csv -m "..."`.
5. To run another variant, change only the `VARIANT = "E4"` line.

Besides `submission.csv`, the notebook writes `ens_checks.json`, `timing_profile.json` and each model's smoothed probabilities (`probs_*_smoothed.npy` with `probs_row_ids.csv`).

## Sharing: what needs doing by hand in the Kaggle UI

I have not changed any visibility. Everything here is **private** under `strippedbanana`. Keep it private, as the project rules require.

Add each teammate as a **collaborator (Viewer)** on each item below. A teammate who can open the E4 notebook but not its sources cannot rerun it.

- **Notebooks:** open the notebook → **Share** (top right) → add the teammate's Kaggle username → Viewer → Save. Do this for:
  1. `sc4000-birdclef-m3-ens-e4`, the final notebook (also `-ens-e2` if you want the fallback)
  2. `sc4000-birdclef-m3-ns-student-main`, the CNN weights
  3. `sc4000-birdclef-m3-a2-a1distill`, the Perch head
  4. `sc4000-birdclef-m3-a2-perch-embed`, the ONNX Runtime wheels
  5. `sc4000-birdclef-m3-ns-student-ablation`. This one is attached to all four ensemble notebooks. Share it, or remove it from the copy's inputs, which is safe for E1/E2/E4.
- **Dataset:** open `strippedbanana/sc4000-birdclef-m3-perch-onnx` → **Settings** → **Sharing** → add collaborator → Viewer.
- Each teammate must also have accepted the BirdCLEF+ 2025 competition rules to attach the competition data.

Not needed for E4: the Bird-MAE weights dataset. It has no declared weights licence and should stay private and unshared.

## How the E4 inputs were trained (for reproduction from scratch; GPU)

- **MAIN student** (`ns-student-main`, T4, ~1.2 h). Trained from the bird-pretrained V2-S backbone (`vladimirsydor/bird-clef-2025-all-pretrained-models`, a public dataset) on `train_audio`, using our team's baseline mel cache (`jmemei/sc4000-birdclef-m1-make-mels`), plus pseudo-labelled `train_soundscapes`.
- **Pseudo-labels** come from `ns-pseudo-labels` v1 (CPU), which reads:
  - `ns-perch-all`: Perch embeddings of all 116,712 soundscape windows;
  - `ns-a1-all`: A1 probabilities, with A1 from `a1-v2s-train`.
  - Teacher = mean(A1, A2 original), then `p^1.3`, then values < 0.01 set to 0. The student checks the pseudo-label files' sha256 before training.
- **A2 + A1-distilled head** (`a2-a1distill`, T4). This is A2's MLP head on Perch embeddings:
  - focal clips come from `a2-perch-embed`;
  - 12,000 `train_soundscapes` windows come from `soundscape-embed`, soft-labelled by A1 (`p^1.3`, floor 0.01, loss weight 0.5).

Reproducing E4 does not require retraining. The weights above are the exact ones that scored.

## Not done

- ONNX/OpenVINO export of the CNN. E4 runs in 32 min on torch CPU, so this was not needed.
- Any ensemble weight beyond the four pre-registered variants. Tuning further on public LB risks overfitting it.
