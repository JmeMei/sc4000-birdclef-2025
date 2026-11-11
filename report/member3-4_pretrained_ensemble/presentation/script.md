# Speaker script: Member 3 (6 min, 9 content slides)

Title slide: 0 s. Opener (optional, one sentence, not counted): We tested whether swapping the pretrained model improves BirdCLEF+ 2025 scores on CPU.

## Slide 1: Problem and constraints

Target time: 41 s (88 words)

BirdCLEF+ 2025 asks for the species present in every five-second window of one-minute soundscapes from Colombia. There are 206 classes: birds, amphibians, insects and mammals, scored by macro ROC-AUC. Training data are mostly clean single-species recordings, but the test data are noisy field recordings, so models must transfer between domains. The submission notebook runs on CPU only, without internet, for at most 90 minutes. We set our own budget of 75 minutes. We used late submissions, which do not count for ranking. Next, the baseline we started from.

Transition: Next, the baseline we started from.

## Slide 2: Baseline v2 and research question

Target time: 39 s (85 words)

Our starting point is baseline v2, from a teammate. It is an EfficientNet-B0 with ImageNet weights, reading log-mel spectrograms. It scored 0.763 public and 0.789 private. The grading cut-off is 0.776 on the public split, and the team goal is 0.810. Our question is narrow. If we keep the training recipe, the split and the inference protocol fixed, and change only the pretrained model, how much does the score improve, and what does it cost in CPU time? That sets up the three model families.

Transition: If we keep the training recipe, the split and the inference protocol fixed, and change only the pretrained model, how much does the score improve, and what does it cost in CPU time? That sets up the three model families.

## Slide 3: Three pretrained families

Target time: 37 s (80 words)

We tried three families. A1 replaces the backbone with a bird-pretrained EfficientNetV2-S from the second-place team. This is a backbone change, not a weights-only swap, and we kept v2's head. A2 uses frozen Perch 2.0 embeddings, 1536 numbers per window, with a small MLP head. A3 uses frozen Bird-MAE-Base features, 768 numbers per window, also with an MLP head. Perch 2.0 was released after the competition closed, so A2 is not comparable to official ranks. Next, how we compared them.

Transition: Next, how we compared them.

## Slide 4: Experimental protocol

Target time: 38 s (82 words)

Every experiment used v2's split: stratified 80/20, seed 42. Each configuration was trained once, with one seed, and submitted once. We treat public differences under 0.01 as noise. Before submitting, a 20-file dry run on unlabelled soundscapes projected the runtime for 700 files, and we submitted only if the projection was at most 75 minutes. There are no labelled soundscapes, so validation uses focal clips only. Validation AUC scores the first five seconds of each clip. Now the results.

Transition: Now the results.

## Slide 5: Results

Target time: 39 s (84 words)

Here are the leaderboard results. A2, Perch with an MLP, is best: 0.84774 public and 0.85226 private. A1 follows with 0.82822 and 0.81642. Both clear the team goal of 0.810 on both splits. A3 scores 0.77640 and 0.78016, which is no reliable gain. Validation AUC was high for all three and understated the gaps, so we choose by leaderboard. A2's validation may be inflated by overlap with Perch's training data. The CPU times for A1, A2 and A3 are projections. Next, a bias finding.

Transition: Next, a bias finding.

## Slide 6: Bias initialisation finding

Target time: 40 s (87 words)

One finding is practical. A linear probe on Bird-MAE features scored 0.7498 with default bias initialisation, and 0.9626 with the bias set to the class log-prior. The cause is likely that a bias starting near zero cannot reach about minus six under a low learning rate. That distance is an analytical estimate, not a measurement. For the MLP, the log-prior did not help: 0.9707 against 0.9749. The Perch logistic regression probably has the same defect, but we did not rerun it. Now, why A3 did not help.

Transition: Now, why A3 did not help.

## Slide 7: Reading A3

Target time: 40 s (87 words)

A3 gives no clear gain. Its public score of 0.77640 clears the 0.776 cut-off by only 0.0004, which is noise. Its private score is 0.009 below v2, and the two splits disagree in sign. We make no claim that it improved. We have three hypotheses, none tested. First, mean pooling discards the structure of the patch map. Second, one pooled vector may represent overlapping species poorly. Third, self-supervised features may encode background as strongly as call identity. We did not run a prototypical probe. Next, inference speed.

Transition: Next, inference speed.

## Slide 8: Inference speed

Target time: 40 s (87 words)

For speed, the fast pipeline uses four workers with one thread each, one mel call per file, and batched windows. Compared like with like, the whole notebook for v2 fell from 11.1 to 5.5 minutes, and A1 takes 8.0. For the pipeline alone, v2 fell from 10.4 to 4.0 minutes, and A1 takes 7.2. These are measured on 700 soundscapes. The changes were bundled, so we cannot attribute the speedup. The submitted notebooks did not use this pipeline, so their times are projections. Last, limitations and conclusions.

Transition: Last, limitations and conclusions.

## Slide 9: Limitations, next steps, conclusion

Target time: 39 s (84 words)

Limitations: one seed, noisy public scores, no labelled soundscapes, and possible validation inflation for A2. A1 also confounds architecture, pretraining and normalisation. Next steps are a noisy-student loop with a strong teacher, a prototypical probe for Bird-MAE, and an ImageNet control for A1. In conclusion, changing only the pretrained model raised the public score from 0.763 to 0.82822 with A1 and 0.84774 with A2. A3 gave no reliable gain. Local validation understated the gaps, and the CPU times for A1 to A3 are projections.

Transition: Local validation understated the gaps, and the CPU times for A1 to A3 are projections.

---

Total words: 764

Total target time: 353 s (5 min 53 s)
