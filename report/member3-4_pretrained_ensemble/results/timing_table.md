# CPU timing: pretrained models and fast inference

Budget: 75 min for 700 one-minute soundscapes (hard limit 90 min). All runs are on Kaggle CPU sessions.

| Pipeline | Model | Backend | Basis | Files | CPU min / 700 | s / file | Measured or projected |
|---|---|---|---|---|---|---|---|
| infer-v2 profile (baseline's pool) | v2 B0 | torch | `train_soundscapes` | 700 | 11.1 whole notebook (10.4 pipeline) | 0.89 | Measured |
| fast-infer v1 (4 workers, 1 thread each) | v2 B0 | torch | `train_soundscapes` | 700 | 5.5 whole notebook (4.0 pipeline) | 0.34 | Measured |
| fast-infer v2 (4 workers, 1 thread each) | A1 V2-S | torch | `train_soundscapes` | 700 | 8.0 whole notebook (7.2 pipeline) | 0.61 | Measured |
| A1 infer v1 (submitted) | A1 V2-S | torch | dry run | 20 | ~37 (~35 + ~2 startup) | 3.0 | Projected, likely pessimistic |
| A2 infer v1 (submitted) | Perch 2.0 + MLP | ONNX Runtime | dry run | 20 | 25.0 | 2.14 | Projected |
| A3 infer v1 (submitted) | Bird-MAE + MLP | torch | dry run | 20 | 61.6 | 5.25 | Projected |

## Stage breakdown

Decode and mel times are summed over workers, so they are CPU time and do not add up to wall time. Model time is summed over workers for the fast and dry-run rows, but is wall time for the profile row, where the model ran in one multi-threaded process.

| Run | Decode (s) | Mel (s) | Model (s) | Setup or model load |
|---|---|---|---|---|
| Profile infer-v2, 700 files (per file) | 0.320 per file | 1.540 per file | 0.412 per file | not separated |
| Fast-infer v1, v2 B0, 700 files | 91.6 | 314.4 | 542.2 | 70.9 s setup |
| Fast-infer v2, A1, 700 files | 73.5 | 240.0 | 1402.7 | 27.7 s setup |
| A1 infer dry run, 20 files (wall) | n/a | 41.1 (features) | 18.7 | not separated |
| A2 infer dry run, 20 files | 4.0 | 0.4 (preprocess) | 163.4 (Perch), 1.44 (head) | 3.8 s per worker |
| A3 infer dry run, 20 files | 3.8 | 100.9 | 313.6 (Bird-MAE), 0.68 (head) | 17.6 s per worker |

## Notes

- The fast pipeline cut v2 from 11.1 to 5.5 min for the whole notebook, and from 10.4 to 4.0 min for the pipeline alone: 4 file-level workers with 1 thread each, one mel call per file on a windows array, and the 12 windows batched.
- The ~25 min reported for our team's baseline was not reproduced. The 11.1 min profile ran the baseline's own pool setup on 700 `train_soundscapes` files. The cause of the difference is unknown, and the baseline's figure may come from the scored run.
- The dry-run projections scale up from 20 files. A1's dry run includes pool startup spread over only 20 files, so it may overstate the 700-file time (startup share not measured).
- All three submitted notebooks scored without a timeout. Kaggle's CLI does not report the scored runtime, so none of these figures is a scored-run measurement.
- ONNX Runtime and OpenVINO runs for the CNNs were not done. In the local smoke test OpenVINO FP16 ran as f32, so it only halved the weight size, with no shown speedup.
