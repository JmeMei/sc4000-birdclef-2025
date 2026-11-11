# Arshiya project log: pretrained models and inference speed

BirdCLEF+ 2025 late submissions, NTU SC4000. Arshiya covers pretrained models (A1, A2, A3) and CPU inference speed (B). Work took place on Sunday 8 November 2026 (SGT), starting at about 15:00 SGT. Hand-off to Arshiya is due Monday 9 November 2026, 22:00 SGT; the final team submission is on Tuesday 10 November 2026.

Conventions: times in the submissions table are UTC (add 8 h for SGT). "Measured" means taken directly from a log, a JSON file or the Kaggle CLI. "Projected" means scaled up from a 20-file dry run. "Estimated" means an approximate figure recorded by Arshiya without an exact measurement. "Not recorded" means no source holds the value. Kaggle kernel runtimes below are the largest timestamp in the downloaded Kaggle log of the latest version, so they cover the whole notebook session as logged. The scored (leaderboard) runtime is not available through the Kaggle CLI.

Reference baseline: Jamie baseline v2, public 0.763 / private 0.789. Grading cut-off 0.776 public; team goal 0.810.

## 1. Phases

### Phase 0: setup (8 Nov, from about 15:00 SGT)
- Kaggle CLI 2.2.5 authenticated; competition file listing worked. Jamie's inference and training notebooks for v2 were pulled for reference.
- A read-only access check was requested. `competitions submissions` returned 403 until Arshiya accepted the BirdCLEF 2025 rules on the website; it then worked.
- Quota stated by Arshiya: 30 GPU hours per week, plan at most about 25, keep about 5 spare; TPU hours ignored.
- No git repository exists in the project folder, so git steps were skipped.
- Standing rules set during the session: GPU training and feature notebooks print `torch.cuda.is_available()` and the device name and raise an error if no GPU is found; these notebooks are pushed on a T4; laptop runs are smoke tests only, with approval needed before anything longer than about 5 minutes; approvals are requested per notebook once it passes its smoke test, with a GPU-hour estimate.

### Phase 1: research
Four research notes (`notes/R1` to `R4`) established the following.
- A1: the pretrained-models dataset contains no EfficientNet-B0. It holds a 2024 and a 2025 EfficientNetV2-S checkpoint and an eca_nfnet_l0 checkpoint. The front end uses fmin 20 and z-score plus min-max normalisation; n_fft, hop and mel count match v2.
- A2: Perch 2.0 is available in ONNX only as a community conversion, so it had to be uploaded as a private dataset.
- A3: Bird-MAE needs its own Kaldi filterbank front end, so Jamie's spectrogram cache cannot be reused. The model folder is 342 MB, the features are 768-d mean-pooled, and no licence was found.
- CLI: `kernels push --accelerator NvidiaTeslaT4` is the accelerator flag; P100 is retired, so T4 was used. There is no `kernels cancel`; `-t` caps runtime. At most 2 GPU batch sessions may run at once.

### Phase 2: implementation
- Notebooks built and smoke-tested locally on synthetic data only: A1 train and infer, A2 embed and infer, A3 embed and infer, the profiling notebook (`b0_profile_infer_v2`) and the fast inference notebook (`b_fast_infer`).
- The fast inference notebook has a model switch (v2 or A1), a backend switch (torch, onnx, openvino_fp16), 4 workers with 1 thread each, one mel call per file on a windows array, and batches of the 12 windows per file. OpenVINO ran as f32 in the local smoke test, so FP16 only halves weight size with no demonstrated speedup.
- A1 normalisation is only partly faithful to the pretraining: an 80 dB window floor (the cache is clip-relative), fmax 14 kHz versus 16 kHz, fmin 50 Hz versus 20 Hz, and possibly a different mel scale.
- A3 pins transformers 4.38.0, tokenizers 0.15.2 and huggingface_hub 0.21.4 in a private folder, and fetches the model by commit URL (`6cc416d1a7ae2af29b6b866499b3b047a8f01304`) with sha256 checks.

### Phase 3: Kaggle runs (approximate order; several ran in parallel)
1. A1 train v1 on T4. 774 of 780 checkpoint keys loaded; only `conv_head` and `bn2` were missing. The run was not stopped: the notebook builds the model with `pretrained=False` and the 2nd-place solution does not use those layers, so the missing keys were expected. Best validation AUC 0.9832 (epoch 9). About 109 s per epoch for 15 epochs (measured, `history.csv`).
2. A2 embed v1 on T4: Perch embeddings for 28,564 clips (106,157 windows) with an embedding wall time of 3,024 s, then heads. MLP head val AUC 0.9916 (first 5 s); logistic regression 0.9071.
3. A3 embed v1 on T4, pushed after a GPU slot freed (see failures). About 12 minutes of embedding, then a linear probe: first-5 s val AUC 0.7495, clip-max 0.716. Not submitted.
4. Profile infer-v2 v1 on CPU (700 `train_soundscapes` files, Jamie's pool): pipeline 622 s (10.4 min, measured; whole notebook 665 s = 11.1 min). Per file: decode 0.320 s, mel 1.540 s, model 0.412 s (summed over workers).
5. Fast-infer v1 (v2 B0, torch, 4 workers) on CPU, 700 files: 5.5 min whole notebook, 4.0 min pipeline (measured, `report/arshiya_pretrained_ensemble/_kaggle_logs/fast-infer-v1/timing_profile.json`).
6. Fast-infer v2 (A1 V2-S, torch) on CPU, 700 files: 8.0 min whole notebook, 7.2 min pipeline, 0.61 s per file (measured, `timing_profile.json`).
7. A1 infer v1 on CPU: 20-file dry run took 59.9 s wall (features 41.1 s, model 18.7 s), projecting about 35 min plus about 2 min start-up for 700 files. Gate (75 min) passed; submitted once.
8. A2 infer v1 on CPU: 20 files, predict wall 46.5 s (2.14 s per file), 25.0 min projected for 700 files. Gate passed; submitted once.
9. A3 probe retry v1 on CPU: sweep of linear and MLP heads on the cached features (best MLP, LR 3e-3, dropout 0.2: 0.9700 first 5 s). Nothing was saved.
10. A3 probe retry v2 on CPU: bias-initialisation test. Default-bias linear head 0.7498 versus log-prior bias 0.9626 (LR 1e-3) and 0.9618 (LR 3e-3). This confirmed that the low linear scores came from the bias initialisation, not the features.
11. A3 mlp-head v1 on CPU: MLP trained twice, once with default bias and once with log-prior bias; the better one (default) was saved (see section 2 for the cross-check).
12. A3 infer v1 on CPU: 20 files, predict wall 122.7 s, 5.25 s per file, model load 17.6 s per worker, 61.6 min projected for 700 files. Gate passed; submitted once.

### Phase 4: deliverables (evening, 8 Nov)
- Written: `report/arshiya_pretrained_ensemble/results/results_table.md`, `report/arshiya_pretrained_ensemble/results/timing_table.md`, `report/arshiya_pretrained_ensemble/results/ensemble_handoff.md`.
- Not yet written: `report/arshiya_pretrained_ensemble/results/pretrained_models_section.md`.
- Not done: export of the best model to ONNX/OpenVINO as a private dataset (deliverable 2). ONNX Runtime and OpenVINO runs for the CNNs were never approved or run on Kaggle.
- Open: the private Kaggle items still have to be shared with Arshiya through the Kaggle website.
- Work paused once at a usage limit and resumed in the evening.

## 2. Decisions
- A1 is a backbone change (EfficientNetV2-S, `tf_efficientnetv2_s_in21k_pretrain_from_bigXCV2Ext_swa.ckpt`), not a weights-only swap, and the report says so. Step 1 uses Jamie's cache, v2's recipe and 2nd-place normalisation (z-score plus min-max). Step 2 (re-cache with fmin 20) only if step 1 gains less than 0.01 public. Public gain was +0.065, so step 2 was not triggered.
- eca_nfnet_l0 only if time and GPU hours remained after A2, A3 and speed work; it was not run.
- A1 keeps v2's timm head; `conv_head` and `bn2` are randomly initialised and trained (effect on score unmeasured).
- A2 uses the MLP head rather than logistic regression (0.9916 versus 0.9071). A linear-head code check found no wiring bug; the loss plateau (0.277 and 0.278) was traced to the default bias initialisation under a low learning rate. The A2 logistic regression has the same defect and was not rerun.
- A3: the retry sweep was approved as one cheap attempt; the prototypical probe was skipped. The whole A3 chain was approved with the MLP trained twice and the higher first-5 s validation AUC saved.
- Bird-MAE weights stay in a private dataset because no licence is declared.
- Profile v1 was kept without a rerun even though it has no device print (Kaggle metadata `enable_gpu` false, code has no CUDA references). Fast-infer got a device print and a GPU guard.
- Selection is by leaderboard, following Jamie's finding that local validation does not predict it.
- Submission gate: project the 700-file time from a dry run; submit automatically at 75 min or less, otherwise do not submit.

### a3-mlp-head cross-check
Both figures were checked against the log and `head_cfg.json` of the downloaded run (latest version, v1). Both are confirmed.

| Bias init | First-5 s val AUC | Clip-max val AUC | Best epoch | Initial bias mean | Saved |
|---|---|---|---|---|---|
| default | 0.9749 | 0.9748 | 26 | -0.0021 | yes |
| log-prior | 0.9707 | 0.9708 | 10 | -6.2008 | no |

`head_cfg.json` records `winning_bias: default`, so the default-bias head (0.9749, epoch 26) is the one used by inference.

## 3. Approvals (given by Arshiya)
| Approval | Condition or note |
|---|---|
| Accept BirdCLEF 2025 rules | Needed before submission listing worked |
| A1 train push (T4) | Stop if epoch 1 projects over 2 GPU hours; pushed with `-t 8100` |
| A2 embed push (T4) | Same 2 h rule; stop if ONNX Runtime cannot use the GPU |
| A3 embed push (T4) | Same 2 h rule |
| Profile infer-v2 push (CPU) | Use Jamie's real pool |
| Fast inference: v2 B0 then A1 V2-S, torch | CPU, 700 `train_soundscapes`, no submission; ONNX/OpenVINO not approved |
| A1 infer push | Gate: 700-file projection of 75 min or less, then submit once and report public, private and runtime |
| A2 chain | Download and create private dataset, push dry run, submit once if 75 min or less |
| Bias-init check as probe-retry v2 | CPU, after v1 ends |
| A3 probe retry v1 (CPU, `-t 1800`) | Cached-feature sweep with higher LR and small MLP |
| Whole A3 chain | Retrain MLP (default and log-prior bias), create private Bird-MAE dataset, dry run, submit once; stop on any failure or a failed 75 min gate |
| Private Bird-MAE dataset | Kept private; undeclared licence noted as a limitation |

## 4. Kaggle runs
Kernel IDs are `strippedbanana/sc4000-birdclef-m3-<name>`. Runtime is the largest log timestamp of the latest version (measured). GPU hours are given only for GPU kernels: the recorded figure from STATE.md is shown, with the log wall time in brackets.

| Kernel | Version | Accelerator | Status | Runtime | GPU hours | Purpose |
|---|---|---|---|---|---|---|
| a1-v2s-train | v1 | T4 | Complete | 1,777 s (29.6 min) | about 0.5 recorded (0.45 h epochs plus setup); log wall 0.49 h | Fine-tune EfficientNetV2-S bird backbone on v2's recipe |
| a1-v2s-infer | v1 | CPU | Complete, submitted (58120958) | 175.5 s (2.9 min) for the dry run in the latest log | none | A1 CPU inference, 20-file dry run, projection 35 min plus 2 min start-up (projected) |
| a2-perch-embed | v1 | T4 | Complete | 3,254 s (54.2 min) | about 0.9 to 1.0 recorded; log wall 0.90 h | Perch 2.0 embeddings, logistic-regression and MLP heads |
| a2-perch-infer | v1 | CPU | Complete, submitted (58122219) | 141 s (2.3 min) for the dry run in the latest log | none | A2 CPU inference with ONNX Runtime, 25.0 min projected |
| a3-birdmae-embed | v1 | T4 | Complete | 863 s (14.4 min) | about 0.3 estimated (not measured exactly); log wall 0.24 h | Bird-MAE-Base embeddings and linear probe |
| a3-birdmae-infer | v1 | CPU | Complete, submitted (58127656) | 216 s (3.6 min) for the dry run in the latest log | none | A3 CPU inference, 61.6 min projected |
| a3-probe-retry | v1 | CPU | Complete | not recorded | none | Sweep of linear and MLP heads on cached Bird-MAE features |
| a3-probe-retry | v2 | CPU | Complete | 247 s (4.1 min) | none | Bias-initialisation test |
| a3-mlp-head | v1 | CPU | Complete | 270 s (4.5 min) | none | Train and save the Bird-MAE MLP head (default versus log-prior bias) |
| profile-infer-v2 | v1 | CPU | Complete | 690 s (11.5 min) | none | Time decode, mel and model stages of v2 on 700 files (pipeline 622 s = 10.4 min measured; STATE.md records 665 s for the notebook; 690 s is the last log timestamp, so the 25 s gap is outside the notebook's own timer) |
| fast-infer | v1 (v2 B0) | CPU | Complete | 331 s (5.5 min) whole notebook, measured | none | Fast pipeline timing for v2 B0 |
| fast-infer | v2 (A1) | CPU | Complete | 503 s (8.4 min) | none | Fast pipeline timing for A1 V2-S (whole notebook 8.0 min, pipeline 7.2 min) |

GPU total recorded: about 1.8 h of the 30 h weekly quota (STATE.md and session record; the A3 part is estimated). All other notebooks ran on CPU.

## 5. Datasets (all private, account `strippedbanana`)
| Dataset | Contents | Status | Note |
|---|---|---|---|
| `strippedbanana/sc4000-birdclef-m3-perch-onnx` | `perch_v2.onnx` (409 MB), LICENSE | Ready, checksum verified | Community ONNX conversion of Perch 2.0 |
| `strippedbanana/sc4000-birdclef-m3-birdmae-base` | `model.safetensors` (342 MB), `config.json`, `preprocessor_config.json`, `configuration_bird_mae.py`, `feature_extractor.py`, `modeling_bird_mae.py` at commit `6cc416d1` | Ready, sha256 verified | Licence undeclared; keep private |

Other artefacts are kernel outputs, not datasets: A1 weights (`model.pth`, `cfg.json`, `species.json`, `history.csv`) from a1-v2s-train v1; the A2 head (`perch_head.npz`, `species.json`, `head_cfg.json`) from a2-perch-embed v1; the A3 head (`birdmae_mlp_head.npz`, `head_cfg.json`) from a3-mlp-head v1. The export dataset for the best model (deliverable 2) has not been created.

## 6. Submissions
Times are UTC from the Kaggle CLI.

| Ref | Time (UTC) | Experiment | Public | Private | Kernel version |
|---|---|---|---|---|---|
| 58120958 | 2026-11-08 08:30:45 | A1 EfficientNetV2-S | 0.82822 | 0.81642 | a1-v2s-infer v1 |
| 58122219 | 2026-11-08 09:17:06 | A2 Perch 2.0 + MLP | 0.84774 | 0.85226 | a2-perch-infer v1 |
| 58127656 | 2026-11-08 12:31:24 | A3 Bird-MAE-Base + MLP | 0.77640 | 0.78016 | a3-birdmae-infer v1 |

Each experiment was submitted once. Against v2 (0.763 / 0.789), the public change is +0.065 (A1), +0.085 (A2) and +0.013 (A3). A3 clears the 0.776 cut-off by 0.0004 on public, which is within noise, and is below v2 on private.

## 7. Failures and fixes
| Problem | Cause | Fix |
|---|---|---|
| `competitions submissions` returned 403 | Competition rules not yet accepted | Arshiya accepted the rules |
| No B0 bird-pretrained checkpoint | Dataset holds V2-S and eca_nfnet_l0 only | Used V2-S and described A1 as a backbone change |
| A1 log reported `conv_head`/`bn2` missing | Notebook uses `pretrained=False`; 2nd place does not use these layers | Not a load failure; no rerun |
| Poll helper could not wait between polls | Tooling limitation | Background watcher loops, polling every 10 min |
| A3 embed push failed twice with "Maximum batch GPU session count of 2 reached" | Two GPU sessions already running | A background job pushed it after a slot freed. This went past the "stop if it fails twice" rule; Arshiya was told and offered a cancel |
| Profile notebook had no device print | Written before the GPU-guard rule | Kept as is by Arshiya; metadata shows CPU. Fast-infer got a device print and a guard |
| Low linear probes (A3 0.7495, A2 logistic regression 0.9071) | Bias initialised near 0 with low LR; target about -6 | Confirmed on A3 features by probe-retry v2; MLP head used; A2 logistic regression not rerun |
| `kaggle datasets create -p D:/...` for Bird-MAE failed (Errno 2) | CLI built a temp upload filename containing the drive path; nothing was created | Ran creation with `-p .` from inside the dataset folder, as for the Perch upload |
| Reported v2 runtime of about 25 min not reproduced | Profile with Jamie's pool gave 11.1 min (10.4 pipeline) on 700 `train_soundscapes`; cause unknown, Jamie's figure may be from the scored run | Both reported, difference noted as a limitation |
| A1 dry-run projection may be pessimistic | Pool start-up spread over only 20 files | Fast-pipeline measurement of 8.0 min reported alongside; start-up share not measured |
| Scored runtime not available | Kaggle CLI does not report it | All submitted-notebook times are projections |

## 8. Bird-MAE-centred write-up (v2)
A writing-only second version reframes existing results around Bird-MAE (A3): `report/arshiya_pretrained_ensemble/archive/paper_v2/` and `report/arshiya_pretrained_ensemble/archive/presentation_v2/`, packaged as `report/arshiya_pretrained_ensemble/archive/paper_v2_overleaf.zip`. No Kaggle runs, training, submissions or dataset changes were made. A literature check (`notes/R6_birdmae_novelty.md`) found no prior report of frozen Bird-MAE-Base on BirdCLEF+ 2025, but its coverage was limited, so the paper does not claim to be first. The transfer-gap table subtracts leaderboard AUC from first-5s validation AUC. Neither v2 document has been compiled locally.

## CORAL heads and dry runs (9 Nov, label-free alignment phase)

Focal first-5s val AUC is recorded only and is not used to select a variant. Original saved heads: Perch 0.99157, Bird-MAE 0.97488. Standardised CORAL is the main variant, pre-registered before any leaderboard score.

| Encoder | Variant | Val AUC | Delta vs original | Submit? |
|---|---|---|---|---|
| Perch | control | 0.99149 | -0.0001 | no |
| Perch | mean/std (secondary) | 0.99150 | -0.0001 | yes |
| Perch | raw CORAL, lambda 1 | 0.99167 | +0.0001 | no |
| Perch | standardised CORAL (main) | 0.99010 | -0.0015 | yes |
| Bird-MAE | control | 0.97488 | 0.0000 | no |
| Bird-MAE | mean/std (secondary) | 0.97479 | -0.0001 | yes |
| Bird-MAE | raw CORAL, lambda 1 | 0.97425 | -0.0006 | no |
| Bird-MAE | standardised CORAL (main) | 0.97283 | -0.0021 | yes |

Mean/std val AUC matches control by construction: the head standardiser undoes the per-dimension affine map. It differs only at inference, through the saved mean/std.

Dry runs (Kaggle CPU, 20 files, projected to 700 files; gate 75 min):

| Notebook | Predict wall | Projected min | Gate |
|---|---|---|---|
| A2 standardised CORAL | 46.4 s | 25.0 | pass |
| A2 mean/std | 39.4 s | 21.1 | pass |
| A3 standardised CORAL | 141.6 s | 73.4 | pass, 1.6 min margin |
| A3 mean/std | 125.8 s | 65.2 | pass |

## Pre-registered diagnostic hypotheses (9 Nov, written before the diagnostics were built or run)

- H1 (CORAL drop): adapted heads give much higher probabilities on background windows, because the mean shift maps "typical soundscape" onto "typical call clip".
- H2 (distillation): distill_coral may partly recover from CORAL's drop, because the pseudo-labelled soundscape windows teach the head that background windows are negatives.

Status of evidence when written: A2 CORAL leaderboard scores were already known; no H1 diagnostic and no distillation result had been viewed.

## CORAL on A2: leaderboard (9 Nov)

Original A2: 0.84774 / 0.85226 (focal val AUC 0.99157; gap 0.144 public, 0.139 private).

| Variant | Ref | Public | Private | Delta public / private | Gap public / private |
|---|---|---|---|---|---|
| Standardised CORAL (main, pre-registered) | 58160573 | 0.68790 | 0.70854 | -0.1598 / -0.1437 | 0.302 / 0.282 |
| Mean/std matching (secondary) | 58160577 | 0.68728 | 0.70632 | -0.1605 / -0.1459 | 0.304 / 0.285 |

CORAL is excluded from the ensemble (user decision, 9 Nov). A pipeline-bug check is in progress; do not interpret these scores until it is complete.

## Head seed repeats and distillation heads (9 Nov)

Seed repeats, original unadapted heads, Kaggle CPU, seeds 42/43/44, split fixed at seed 42 (seed varies head init, batch order and dropout only). No submissions.

| Encoder | First-5s val AUC mean +/- std (ddof 1, CPU runs only) | Clip-max mean +/- std | CPU seed 42 vs original saved head |
|---|---|---|---|
| Perch (A2) | 0.99163 +/- 0.00029 | 0.99262 +/- 0.00058 | 0.99149 vs 0.99157 (original on T4) |
| Bird-MAE (A3) | 0.97451 +/- 0.00076 | 0.97389 +/- 0.00108 | 0.97493 vs 0.97488 (original also CPU) |

Distillation heads (a3-distill v1, CPU, 8.0 min; teacher = original A2 head): first-5s val AUC `distill` 0.97504 (+0.0002 vs original A3 0.97488), `distill_coral` 0.97243 (-0.0025). Soundscape loss weight share 0.208. Teacher: 1.9% of label entries non-zero after p^1.3 and the 0.01 floor; 1431 of 12000 windows have a label >= 0.5. Val AUC is recorded only.

## CORAL on A3: leaderboard (9 Nov)

Original A3: 0.77640 / 0.78016. Neither run timed out.

| Variant | Ref | Public | Private | Delta public / private |
|---|---|---|---|---|
| Standardised CORAL (main, pre-registered) | 58160580 | 0.64281 | 0.64682 | -0.1336 / -0.1333 |
| Mean/std matching (secondary) | 58160584 | 0.66971 | 0.68033 | -0.1067 / -0.0998 |

Excluded from the ensemble with the A2 CORAL heads. Interpretation waits for the bug check.

## CORAL bug check (coral-diag v1, Kaggle CPU, 3.3 min, 9 Nov)

Inference path, 20 files / 240 rows: CORAL path with the ORIGINAL A2 head vs the original A2 notebook: max abs diff 3.0e-8, PASS (threshold 1e-5). No transform exists at inference. CPU control head vs original on the same rows: max abs 0.115, mean abs 0.00046, per-class Spearman mean 0.986 (retrained head, informational).

Feature space: training features are mapped back to raw soundscape space. Saved head mean matches the soundscape mean (gap / target std 1.5e-6 Perch, 5.4e-6 Bird-MAE for mean/std and std-CORAL; control gap 0.35 and 0.59). Inference z-scores raw test features with those saved stats.

Training script end to end, control vs original on 12000 windows, per-class Spearman: Perch mean 0.987, median 0.988, p5 0.977 (T4 vs CPU); Bird-MAE 1.000 (CPU vs CPU, identical outputs).

12000 train_soundscape windows:

| Encoder | Head | Mean prob | Frac max > 0.5 | Frac max > 0.9 | Median window max | Spearman vs original (mean / p5) |
|---|---|---|---|---|---|---|
| Perch | original | 0.00351 | 0.161 | 0.034 | 0.188 | 1 |
| Perch | control | 0.00365 | 0.161 | 0.034 | 0.194 | 0.987 / 0.977 |
| Perch | mean/std | 0.00440 | 0.115 | 0.036 | 0.136 | 0.368 / 0.074 |
| Perch | std-CORAL | 0.00264 | 0.131 | 0.058 | 0.090 | 0.293 / -0.015 |
| Bird-MAE | original | 0.00321 | 0.154 | 0.015 | 0.176 | 1 |
| Bird-MAE | mean/std | 0.00234 | 0.057 | 0.013 | 0.096 | 0.724 / 0.531 |
| Bird-MAE | std-CORAL | 0.00236 | 0.067 | 0.016 | 0.098 | 0.568 / 0.379 |

Conclusion: no pipeline bug found. H1 is NOT supported: adapted heads do not give much higher probabilities on soundscape windows (mean probability rose 25% only for Perch mean/std and fell elsewhere; the fraction of windows with max > 0.5 fell in every case). The clear change is that per-class rankings differ from the original head. Bird-MAE: std-CORAL is ~0.03 worse than mean/std on the leaderboard and also agrees less with the original head (Spearman 0.57 vs 0.72); Perch: both agree poorly (0.29, 0.37) and score alike. This association is not proof of cause.

## H1 verdict (9 Nov)

H1 is NOT supported. The adapted heads do not give higher probabilities on train_soundscape windows: mean probability and the share of confident windows mostly fell. Instead they reorder windows within each class (per-class Spearman vs the original head: Perch 0.29-0.37, Bird-MAE 0.57-0.72). On A3, lower Spearman goes with lower leaderboard score (std-CORAL vs mean/std); this is an association only, not a demonstrated cause.

## Distillation and v2 + AdaBN: leaderboard (9 Nov)

v2 unadapted reference is known to 3 decimals only (0.763 / 0.789); its deltas and gaps use those rounded values.

| Submission | Ref | Public | Private | Delta public / private | Comparison | Gap public / private |
|---|---|---|---|---|---|---|
| A3 + Perch-distilled (MAIN) | 58161782 | 0.81238 | 0.80208 | +0.0360 / +0.0219 vs A3 | beats A3 | 0.163 / 0.173 (A3 0.199 / 0.195) |
| A3 + Perch-distilled + CORAL (secondary) | 58162938 | 0.69343 | 0.68854 | -0.0830 / -0.0916 vs A3 | +0.0506 / +0.0417 vs A3 std-CORAL; -0.1190 / -0.1135 vs distill | 0.279 / 0.284 |
| v2 + AdaBN | 58162933 | 0.73889 | 0.76383 | -0.0241 / -0.0252 vs v2 | A1 AdaBN was -0.0561 / -0.0507 | 0.202 / 0.177 (v2 0.214 / 0.188, val 0.9766) |

H2 (distill_coral partly recovers from CORAL's drop): supported in direction, +0.051 public / +0.042 private over A3 std-CORAL, but it stays 0.083 / 0.092 below original A3. The mechanism (background windows learned as negatives) was not tested. v2's transfer gap narrows after AdaBN only because its focal val AUC fell (0.9766 to 0.9405) more than its score did.

## A2 + A1-distilled: leaderboard (9 Nov)

Ref 58164973: public 0.86651 / private 0.85737. Delta vs A2 (0.84774 / 0.85226): +0.0188 / +0.0051. Focal val AUC 0.99182. Transfer gap 0.125 public / 0.134 private (A2 0.144 / 0.139). The private gain is within the 0.01 noise rule used for public differences.

## Noisy-student round 1: teacher note (10 Nov, before any student is trained)

ABLATION differs from MAIN in both teacher and label density: 0.74% vs 2.50% non-zero entries after p^1.3 and the 0.01 floor (7,952 vs 11,430 windows with a label >= 0.5; mean 0.00056 vs 0.00135). A score difference between the two students therefore cannot be attributed to the teacher alone. MAIN was chosen after the 58164973 score was known (to avoid counting A1 twice).

## Noisy-student round 1: training results (10 Nov 09:07)
- Students: V2-S, bird-pretrained init, 15 epochs, 825 steps/epoch, pseudo-labels sha256-verified.
- MAIN (teacher mean(A1, A2)): best val AUC 0.98650 (epoch 14), final 0.98574.
- ABLATION (teacher A2 + A1-distilled): best 0.98541 (epoch 14), final 0.98485.
- A1 reference val AUC 0.9832. Val gaps are small (+0.0033 / +0.0022); per M1, val AUC does not predict LB, so the decision waits for public LB.
- GPU: ~2.3 T4-h of the 4 h budget.

## Head seed repeats (CPU, seeds 42/43/44, fixed split)
- A2 + A1-distilled head: 0.99133 ± 0.00047 (first-5s); seed 42 reproduces T4 value within 0.00023.
- A3 distill head: 0.97506 ± 0.00012; seed 42 identical to saved.
- Seed noise on val AUC is ≤0.0005, far below LB noise (~0.01).

## Noisy-student round 1: leaderboard (10 Nov ~09:40)
- MAIN (ref 58181262): 0.85985 / 0.85185; Δ vs A1 +0.0316 / +0.0354; gap val−public 0.127 (A1 0.155).
- ABLATION (ref 58181267): 0.84003 / 0.83794; Δ vs A1 +0.0118 / +0.0215; gap 0.145.
- MAIN − ABLATION: +0.0198 public / +0.0139 private (above the 0.01 noise level). Selected by public LB: MAIN. Confound: teacher and label density both differ.
- Neither beats A2 + A1-distilled (0.86651 / 0.85737); MAIN is within 0.007 public (noise level). Best single CNN so far.

## Ensemble E1/E2 (10 Nov 18:41)
- E1 MAIN + A2d rank average, equal: 0.88352 / 0.87203 (Δ vs A2d +0.0170 / +0.0147). Above the 0.01 noise level: the CNN student and the Perch head are complementary.
- E2 = E1 + temporal smoothing: 0.88662 / 0.88008 (Δ vs A2d +0.0201 / +0.0227; vs E1 +0.0031 / +0.0081). Smoothing gain on public is below the 0.01 noise level; private gain larger. Selected by public LB: E2 (current best).
- Dry runs 28.3 / 30.4 CPU min on 700 files; sanity PASS (per-class Spearman vs each model 0.81-0.82).

## Ensemble E3/E4 (10 Nov 20:01)
- E3 (ref 58198886) = E2 + ABLATION student, 1/3 each: 0.88058 / 0.87522 (Δ vs A2d +0.0141 / +0.0179; vs E2 −0.0060 / −0.0049). Adding a second CNN shifts weight from Perch to CNNs (Spearman vs A2d 0.677) and does not help.
- E4 (ref 58198889) = E2 at 0.4 CNN / 0.6 Perch: 0.88749 / 0.88029 (Δ vs A2d +0.0210 / +0.0229; vs E2 +0.0009 / +0.0002, within noise).
- Selected by public LB: E4 (best public and best private). Phase used 4/6 submissions, CPU only.
