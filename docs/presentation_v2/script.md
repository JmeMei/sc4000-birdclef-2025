# Speaker script: Bird-MAE talk (v2)

## Slide 1: Question: do SSL foundation models transfer?

Target time: 58 s (125 words)

Foundation models promise a simple recipe: pretrain once on large audio collections, freeze the encoder, and train only a small head. Self-supervised learning is attractive in bioacoustics, because unlabelled recordings are plentiful, while species labels, especially for continuous soundscapes, are expensive. Bird-MAE is a masked autoencoder pretrained on bird recordings. Our question is whether its frozen features transfer to field soundscapes when the head sees only clean focal clips. We treat this as a diagnostic study, not a bid to win. We asked what the number means. In a limited search, we found no prior report of frozen Bird-MAE-Base on this competition. The practical promise is that a frozen encoder and a light head could replace expensive end-to-end training. Next, the task and its constraints.

Transition: Next, the task and its constraints.

## Slide 2: Task and constraints

Target time: 56 s (121 words)

BirdCLEF+ 2025 asks for the species present in every five-second window of one-minute soundscapes from Colombia. There are 206 classes: 146 birds, 34 amphibians, 17 insects and nine mammals. The metric is macro ROC-AUC, skipping classes with no positives. Training data are mostly short single-species focal clips, while test data are continuous field recordings with wind, rain and overlapping calls. The submission notebook runs on CPU only, with internet off, for at most 90 minutes; our own budget is 75. We used late submissions, which are not ranked. Our baseline, v2, is an EfficientNet-B0 that scored 0.763 public and 0.789 private. A model that separates species well on clean clips may not do so under these conditions. Next, the Bird-MAE pipeline.

Transition: Next, the Bird-MAE pipeline.

## Slide 3: Method: Bird-MAE pipeline

Target time: 56 s (121 words)

Audio is cut into five-second windows at 32 kilohertz. Bird-MAE's own feature extractor builds a 128-bin filterbank of 512 frames, which gives 256 patches. The encoder is a ViT-B/16 with 85.5 million parameters, and it stays frozen. We take the mean of the patch tokens, a 768-dimensional vector per window, and cache it to disk once. That took about 0.3 GPU hours on a T4. Two heads are then trained from the cache: a linear layer, and an MLP with 512 hidden units and dropout 0.2. Targets are 1.0 for primary and 0.5 for secondary labels, on v2's split. We called the extractor once per window, so each input is independent of its neighbours. Next, the first result, which looked bad.

Transition: Next, the first result, which looked bad.

## Slide 4: A linear probe that looked poor

Target time: 57 s (123 words)

The first linear probe reached 0.7495 validation AUC, and a retry gave 0.7498. The usual reading is that pooled features are a weak interface, which would suggest poor features. But the training loss told a different story. It started at 0.6929, close to ln 2, the loss of predicting probability one half everywhere. It then fell slowly to a plateau of 0.277. A head that had merely learned the class base rates would reach a far lower loss. So the plateau says the output offset had not been learned; it says little about the features themselves. That made us suspect the optimiser and the bias. Nothing here says the features are bad; the number simply needs explaining. Next, the sweep that pointed there.

Transition: Next, the sweep that pointed there.

## Slide 5: Sweep and bias diagnosis

Target time: 56 s (122 words)

Raising the linear learning rate to 0.003 and 0.01 lifted AUC to 0.8513 and 0.9370, so step size mattered. The default bias starts near zero. A uniform-class estimate puts the right value near minus 5.3, and the measured mean log-prior is minus 6.20. A code review estimated that at learning rate 0.001 an Adam-style update moves a bias by only about 1.3 in total. That is an analytical estimate, not a measurement. Changing only the bias to the log-prior gave 0.9626, and the first-epoch loss fell from 0.6941 to 0.0285. This is one seed, so it supports the diagnosis without proving it. The same features, the same data and the same learning rate gave two very different answers. Next, the MLP heads.

Transition: Next, the MLP heads.

## Slide 6: MLP heads

Target time: 53 s (115 words)

An MLP can supply its own offset. With 12 epochs it reached 0.9700, and with 30 epochs 0.9749, a gain of 0.0049. We then tried the log-prior bias on the MLP. It reached 0.9707, which is 0.0042 lower than the default. So the fix that rescued the linear probe did not help the MLP. The default-bias head won on validation, and it is the one we submitted. One possible explanation is that the hidden layer can supply a large negative offset, but we did not test it. Epoch counts differ between the sweep rows, so read that comparison with care. Retraining both heads on CPU took under two minutes each. Next, what the leaderboard said.

Transition: Next, what the leaderboard said.

## Slide 7: Leaderboard

Target time: 48 s (105 words)

We submitted the default-bias MLP once. It scored 0.77640 public and 0.78016 private. The public gain over v2 is plus 0.013, only slightly above our 0.01 noise threshold. It clears the 0.776 cut-off by 0.0004, which is noise. The private score is 0.009 below v2, so we claim no improvement. For reference, A1, the bird-pretrained EfficientNetV2-S, scored 0.82822 and 0.81642. A2, frozen Perch 2.0 with an MLP head, scored 0.84774 and 0.85226. Perch 2.0 was released after the competition closed, so A2 is not comparable to official ranks. No ensemble was tested, and we make no ranking claim. Next, the gap between validation and leaderboard.

Transition: Next, the gap between validation and leaderboard.

## Slide 8: Transfer gap

Target time: 52 s (113 words)

Subtracting leaderboard AUC from validation AUC gives the transfer gap. For Bird-MAE it is 0.1985 on the public split, against 0.1439 for Perch and 0.1550 for the CNN. Validation spans only 0.017 across the three models, while public scores span 0.071. The caveats matter. This is a single seed and a single split. Validation covers focal clips and the leaderboard covers soundscapes with a different class set, so the gap mixes domain shift with class coverage. A2's validation may also be inflated by data overlap. We read the gap as descriptive only. Bird-MAE's gap was the largest on both splits, 0.1947 on the private one. Next, three hypotheses for why Bird-MAE transferred worse.

Transition: Next, three hypotheses for why Bird-MAE transferred worse.

## Slide 9: Why? Three untested hypotheses

Target time: 55 s (119 words)

We offer three hypotheses, and none has been tested. H1: mean pooling dilutes short calls. The test is to replace it with attention or max pooling, or a prototypical probe, over cached patch tokens. H2: the head saw only clean-clip features. The test is to embed focal clips with soundscape noise mixed in, and retrain. But A2 was also trained on clean features and transferred best, so H2 alone cannot explain the difference. H3: the pretraining objective and data differ, label-free versus supervised. The test needs a labelled soundscape subset for per-taxon AUC. The hypotheses may also interact. Each hypothesis also has a reason it might be wrong, which we give in the paper. Next, cost and practical lessons.

Transition: Next, cost and practical lessons.

## Slide 10: Cost and practical lessons

Target time: 55 s (120 words)

Embedding 106,157 windows took 691 seconds on a T4, about 0.3 GPU hours by our estimate. CPU inference was the slowest of our three models. A 20-file dry run took 5.25 seconds per file with four workers, which projects to 61.6 minutes for 700 files. That is a projection, not a scored-run measurement. Perch projects to 25.0 minutes and A1 to about 37, so Bird-MAE leaves the least headroom for an ensemble. Two lessons stand out. First, check the first-epoch loss against ln 2. Second, do not rank encoders by clean validation AUC, which compressed a 0.071 public spread into 0.017. Summed over workers, the encoder took 313.6 seconds against 100.9 for the mel front end. Next, limitations and conclusion.

Transition: Next, limitations and conclusion.

## Slide 11: Limitations, future work, conclusion

Target time: 59 s (128 words)

The evidence rests on one seed, one split and one submission per model, so there are no error bars. The hypotheses are untested, the only causal result is the bias one, and we did not run a prototypical probe. The Perch logistic regression plateaued at 0.278, which we presume has the same defect, but we did not re-test it. Future work follows the hypotheses: a pooling ablation, noise-augmented embeddings and per-taxon evaluation. In conclusion, frozen Bird-MAE with an MLP scored 0.77640 public and 0.78016 private, no clear improvement on v2 and well below Perch. A linear probe at 0.75 reached 0.96 once its bias was corrected, so check the diagnosis before judging features. The Bird-MAE weights carry no declared licence, so we kept them private. Thank you; we welcome questions.

Transition: Thank you; we welcome questions.

Total: 1312 words, about 10.1 min at 130 words/min.
