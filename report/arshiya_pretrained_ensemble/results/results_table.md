# Results: pretrained models, transfer and ensemble

All runs use v2's split (seed 42, stratified 80/20). Each was submitted once as a late submission.
Reference points: v2 public 0.763 / private 0.789; grading cut-off 0.776 public; team goal 0.810.

| Experiment | Val AUC | Public | Private | Δ public vs v2 | Δ private vs v2 | CPU min / 700 files |
|---|---|---|---|---|---|---|
| v2 baseline (team baseline) | not recorded | 0.763 | 0.789 | 0 | 0 | 11.1 measured (profile); 5.5 measured (fast pipeline) |
| A1 EfficientNetV2-S, bird-pretrained | 0.9832 | 0.82822 | 0.81642 | +0.065 | +0.027 | ~37 projected (submitted notebook); 8.0 measured (fast pipeline) |
| **A2 Perch 2.0 + MLP head** | 0.9916 | **0.84774** | **0.85226** | +0.085 | +0.063 | 25.0 projected |
| A3 Bird-MAE-Base + MLP head | 0.9749 | 0.77640 | 0.78016 | +0.013 | -0.009 | 61.6 projected |

Val AUC is on the first 5 s of each validation clip. Local val AUC did not rank the models the same way as the private leaderboard in our team's baseline work, so we choose by leaderboard.

Submission refs: A1 58120958, A2 58122219, A3 58127656.

## Reading the results

A2 is the best on both public and private, and A1 is second on both. Both clear the 0.810 team goal on both splits.

A3 clears the 0.776 cut-off by only 0.0004 on public, which is within noise (public differences under 0.01), and is below v2 on private.

A3's val AUC (0.9749) sits close to A2's (0.9916), but its leaderboard scores are far lower. This is another case of local validation not predicting the leaderboard.

## Linear and alternative heads (validation only, not submitted)

| Features | Head | First-5s val AUC | Clip-max val AUC |
|---|---|---|---|
| Perch 2.0 | Logistic regression, default bias | 0.9071 | 0.9113 |
| Bird-MAE | Linear, LR 1e-3, default bias | 0.7498 | 0.7183 |
| Bird-MAE | Linear, LR 1e-3, log-prior bias | 0.9626 | 0.9636 |
| Bird-MAE | Linear, LR 3e-3, log-prior bias | 0.9618 | 0.9631 |
| Bird-MAE | MLP, default bias (saved, submitted) | 0.9749 | 0.9748 |
| Bird-MAE | MLP, log-prior bias | 0.9707 | 0.9708 |

The low default-bias linear scores came from bias initialisation under a low learning rate, not from the features. The final bias starts near 0 and needs about -6. The Perch logistic regression has the same defect and was not rerun.

## Caveats

- A1 is a backbone change from v2's B0, not a weights-only swap. It keeps v2's head, not the 2nd-place attention head. Its normalisation only partly matches the pretraining (80 dB floor, fmax 14 kHz vs 16 kHz, fmin 50 Hz vs 20 Hz, possibly different mel scale).
- Perch 2.0 was released after the competition closed, and was trained on Xeno-canto and iNaturalist. A2's val AUC may be inflated by overlap with `train_audio`, so the leaderboard is the fair test. The ONNX model is a community conversion.
- Bird-MAE declares no licence for its weights on the Hugging Face card, GitHub repo or arXiv listing. The weights dataset stays private.
- A3 used a mean-pooled MLP head. The prototypical probe, which the Bird-MAE paper reports as much stronger than linear probing, was not run.

## Label-free statistics alignment (train_soundscapes only)

Adaptation variants were pre-registered before any leaderboard score was seen. Focal val AUC is recorded but not used for selection. Transfer gap = focal val AUC minus leaderboard score.

| Model | Adaptation | Val AUC | Public | Private | Δ public / private vs unadapted | CPU min / 700 files | Transfer gap before → after (public; private) |
|---|---|---|---|---|---|---|---|
| A1 | AdaBN, 12,000 windows from 1000 train_soundscapes (ref 58158788) | 0.9713 (unadapted 0.9832) | 0.77210 | 0.76573 | −0.0561 / −0.0507 | 10.7 measured | 0.155 → 0.199; 0.167 → 0.206 |
| A2 | Standardised CORAL, main (ref 58160573) | 0.9901 (unadapted 0.9916) | 0.68790 | 0.70854 | -0.1598 / -0.1437 | 25.0 projected | 0.144 -> 0.302; 0.139 -> 0.282 |
| A2 | Mean/std matching, secondary (ref 58160577) | 0.9915 | 0.68728 | 0.70632 | -0.1605 / -0.1459 | 21.1 projected | 0.144 -> 0.304; 0.139 -> 0.285 |
| A3 | Standardised CORAL, main (ref 58160580) | 0.9728 (unadapted 0.9749) | 0.64281 | 0.64682 | -0.1336 / -0.1333 | 73.4 projected | 0.199 -> 0.330; 0.195 -> 0.326 |
| A3 | Mean/std matching, secondary (ref 58160584) | 0.9748 | 0.66971 | 0.68033 | -0.1067 / -0.0998 | 65.2 projected | 0.199 -> 0.305; 0.195 -> 0.295 |
| A3 | Perch-distilled, main (ref 58161782) | 0.9750 (unadapted 0.9749) | 0.81238 | 0.80208 | +0.0360 / +0.0219 | 67.5 projected | 0.199 -> 0.163; 0.195 -> 0.173 |
| A3 | Perch-distilled + std-CORAL source, secondary (ref 58162938) | 0.9724 | 0.69343 | 0.68854 | -0.0830 / -0.0916 | 61.9 projected | 0.199 -> 0.279; 0.195 -> 0.284 |
| v2 (B0) | AdaBN, 12,000 windows (ref 58162933) | 0.9405 (unadapted 0.9766) | 0.73889 | 0.76383 | -0.0241 / -0.0252 | 4.7 measured | 0.214 -> 0.202; 0.188 -> 0.177 |
| A2 | A1-distilled (ref 58164973) | 0.9918 (unadapted 0.9916) | 0.86651 | 0.85737 | +0.0188 / +0.0051 | 26.5 projected | 0.144 -> 0.125; 0.139 -> 0.134 |

Control: unadapted A1 scored through the same fast-inference notebook (ref 58159079) gave 0.82822 / 0.81642, identical to the per-window A1 notebook, so the A1 + AdaBN drop is due to AdaBN and not the inference path. The control was requested after the A1 + AdaBN score was seen.

## Noisy-student round 1 (A1 student on pseudo-labelled train_soundscapes)

| Base | Teacher (ref) | Val AUC | Public | Private | Δ vs A1 (public / private) | CPU min / 700 files | Gap val−public (A1 0.155) |
|---|---|---|---|---|---|---|---|
| A1 V2-S | MAIN: mean(A1, A2) (ref 58181262) | 0.9865 | 0.85985 | 0.85185 | +0.0316 / +0.0354 | 9.9 measured | 0.127 |
| A1 V2-S | ABLATION: A2 + A1-distilled (ref 58181267) | 0.9854 | 0.84003 | 0.83794 | +0.0118 / +0.0215 | 10.0 measured | 0.145 |

ABLATION differs from MAIN in both teacher and label density (0.74% vs 2.50% non-zero), so the MAIN−ABLATION difference cannot be attributed to the teacher alone.

## Final ensemble: per-class rank average, CPU, one shared decode

Pre-registered 10 Nov 09:55 SGT before any ensemble result. Δ vs best single model A2 + A1-distilled (0.86651 / 0.85737).

| Variant | Models (weights) | Smoothing | Public | Private | Δ vs A2d (public / private) | CPU min / 700 files |
|---|---|---|---|---|---|---|
| E1 (ref 58196484) | MAIN student 0.5 + A2d 0.5 | no | 0.88352 | 0.87203 | +0.0170 / +0.0147 | 28.3 measured |
| E2 (ref 58196486) | MAIN student 0.5 + A2d 0.5 | 0.25/0.5/0.25 | 0.88662 | 0.88008 | +0.0201 / +0.0227 | 30.4 measured |
| E3 (ref 58198886) | MAIN 1/3 + ABLATION 1/3 + A2d 1/3 | 0.25/0.5/0.25 | 0.88058 | 0.87522 | +0.0141 / +0.0179 | 39.9 measured |
| **E4 (ref 58198889), selected** | MAIN 0.4 + A2d 0.6 | 0.25/0.5/0.25 | **0.88749** | **0.88029** | **+0.0210 / +0.0229** | 32.3 measured |

Selection by public LB (pre-registered): E4. E4 vs E2 differs by +0.0009 / +0.0002, well below the 0.01 noise level, so the two are equivalent; E3 (adding the ABLATION student) is lower than E2 on both splits.
