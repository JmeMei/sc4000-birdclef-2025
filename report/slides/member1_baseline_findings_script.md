# Member 1 – Data findings and baseline: video script

Slides: `member1_baseline_findings.pptx` (5 slides). Target length: **about 2 min 30 s**,
which leaves about 7.5 min for the other three members in the 10-minute video.
The same text is in each slide's speaker notes.

| Slide | Title | Time |
|---|---|---|
| 1 | Data findings and baseline | 0:05 |
| 2 | A few species have most of the data | 0:35 |
| 3 | Four problems the data showed us | 0:40 |
| 4 | Baseline: treat sound as an image | 0:35 |
| 5 | Result: rank about 1600 of 2032 | 0:35 |
| | **Total** | **≈ 2:30** |

---

### Slide 1 – Data findings and baseline (≈ 5 s)

Hi, I'm Member 1. I'll cover what we learned from the data, the baseline model we built,
and how it scored.

### Slide 2 – A few species have most of the data (≈ 35 s)

The training set has 28,564 short recordings of 206 species: mostly birds, plus amphibians,
insects and mammals. The data is very uneven. The most common species has 990 clips, the
median is 80, and 30 species have five clips or fewer. That matters because the score is
macro ROC-AUC, so a rare species counts as much as a common one.

### Slide 3 – Four problems the data showed us (≈ 40 s)

Exploring the data showed four problems.
One, class imbalance, as we just saw.
Two, weak labels: one label covers a whole recording of about 20 seconds, but we must predict
every 5-second window, and the bird may only call for part of it.
Three, domain shift: training clips are clean, focused recordings, but the test set is noisy
one-minute soundscapes.
Four, missing labels: other species calling in the background are rarely labelled.
On top of that, the submission must run on CPU, offline, in under 90 minutes.
These four problems shaped everything we tried next.

### Slide 4 – Baseline: treat sound as an image (≈ 35 s)

Our baseline treats sound as an image. We take the first 5 seconds of each clip, turn it into
a log-mel spectrogram, which shows frequency over time, and feed it to EfficientNet-B0
pretrained on ImageNet. It outputs one probability for each of the 206 species.
We kept it simple on purpose: primary label only, BCE loss, no augmentation, 15 epochs.
Validation AUC peaked at 0.973 at epoch 7, then the model started to overfit.

### Slide 5 – Result: rank about 1600 of 2032 (≈ 35 s)

On the leaderboard the baseline scored 0.717 public, 0.699 private, around rank 1600 of
2032, or the top 79 percent. That is far below our local 0.973, because our validation clips
come from the same clean recordings as training, while the test set is noisy soundscapes.
So local validation overstates performance and we judge models on the leaderboard.
To reach the next grading band we need 0.776, and the rest of our work targets the four
problems I just described.

---

**Tips for recording**
- Speak to the numbers on screen; you don't need to read the slide text aloud.
- If you run long, shorten slide 3: name the four problems in one line each.
