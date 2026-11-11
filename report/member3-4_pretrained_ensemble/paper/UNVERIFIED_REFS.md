# Unverified or limited references

Sources: `notes/R7_paper_citations.md`, `notes/R7_followup.md` (checked 2026-11-10).

## Not cited on purpose
- Pseudo-Label (Lee, 2013), `lee2013pseudo`: not cited. Workshop paper by Dong-Hyun Lee that trains a network on its own most-confident predictions for unlabeled data as if they were labels. The venue is unverified (search summaries say ICML 2013 Workshop on Challenges in Representation Learning; the PDF was unreadable and no workshop page was found). The paper cites Xie et al. (2020) for pseudo-labeling instead.
- AdaBN arXiv version (`li2016revisiting`): not cited. The paper cites the journal version `li2018adaptive` (Pattern Recognition 80:109-117), whose title is "Adaptive Batch Normalization for practical domain adaptation" and differs from the arXiv title. The ICLR 2017 workshop claim is unverified.
- LifeCLEF 2025 overview (`picek2026lifeclef`): unverified (HAL snippet only), not cited.

## Cited with limits
- BirdCLEF+ 2026 working notes `chen2026class` (paper 330), `farazi2026crossmutated` (paper 340), `banthia2026pantanal` (paper 323): the PDFs could not be read by our tools. The paper describes them only from abstracts and search summaries: 330 as a sound-event-detection pipeline with semi-supervised noisy-student distillation; 340 as iterative semi-supervised distillation with a Perch-based ProtoSSM branch and per-class rank blending; 323 by its title (runtime-constrained distillation, diversity-aware ensembling). Update 10 Nov 2026: the authors verified paper 340 themselves; it proposes cross-mutation distillation (pseudo-label teachers from a different backbone lineage than the student, against same-family self-training saturation). The paper now cites it for this in Related Work and Discussion. Cross-family distillation in 330 remains unconfirmed and is not claimed. Teacher of 323 is not stated.
- `kahl2026overview` (BirdCLEF+ 2026 overview, paper 317): verified on the CEUR index; author list truncated with "and others".
- Sydorskyi and Goncalves (`sydorskyi2025tackling`): the system description (strong baseline, in-domain transfer learning, semi-supervised distillation, postprocessing, ensembling) comes from a non-CEUR listing and a search summary; the CEUR PDF text was not read. Their reported scores are not used; the paper uses the Kaggle final leaderboard instead.
- Leaderboard figures (1st 0.93067, 2nd 0.92893): read with the Kaggle CLI on 10 Nov 2026 (`report/member3-4_pretrained_ensemble/_kaggle_logs/_writeup/leaderboard_top.txt`).
- DS@GT scores (`miyaguchi2025distilling`: best model-zoo model BirdSetEfficientNetB1, 0.778 private, 0.810 public): from arXiv 2507.08236 (abstract, conclusion, Table 11). Scores for their token method are unresolved and not cited.
- Boudiaf et al. (`boudiaf2023search`, ICML 2023): verified; abstract states evaluation on naturally occurring distribution shifts in bioacoustics. It does not say birds or audio explicitly.
- BirdSet (`rauch2025birdset`): ICLR 2025 verified on the ICLR proceedings page. Spotlight status (arXiv comment only) is not stated in the paper.
- Bird-MAE (`rauch2025can`): TMLR 2025; no volume or pages. The 37-point figure is from the arXiv abstract.
- Perch 2.0 (`vanmerrienboer2025perch`): arXiv only, no venue. Release 6 Aug 2025 (v1); competition deadline 5 Jun 2025 (imageclef.org/BirdCLEF2025).
- mixup (`zhang2018mixup`): kept. The arXiv page (1710.09412, v2 of 27 Apr 2018) gives title and authors and the comment "ICLR camera ready version"; the year 2018 is inferred from that date plus the secondary mlanthology listing, as there is no journal-ref field.
- Leaderboard 2nd place: the project brief said 0.928; the Kaggle final leaderboard lists 0.92893 (0.929 rounded) for the second team ("volodymyr vialactea"), so the paper uses 0.92893. That this team is the Sydorskyi and Goncalves paper is inferred from the author's first name and the paper's 2nd-place claim, not from a team page.
- The Kaggle CLI leaderboard output does not label itself public or private; for a closed competition it is the final standing, which Kaggle ranks by the private score.
- SpecAugment (`park2019specaugment`): cited only to say it was not used (Interspeech 2019, pages and DOI from R5).
- Tent (`wang2021tent`): ICLR 2021 confirmed on the ICLR site via search; no pages. Cited as related work only, not tested.
- Audio-MAE (`huang2022masked`): NeurIPS 35; no page numbers. MAE (`he2022masked`): CVPR 2022, pages from Crossref.
- Noisy Student, stochastic depth, focal loss, EfficientNet, EfficientNetV2, CORAL, Hinton et al., BirdNET: as in R5/R7 (Crossref or proceedings pages).

## Not sourced in the paper
- Ghani et al. (global bird embeddings) and Schwinger et al. (foundation-model review) were in the earlier draft but are not in the verified bibliography, so they are no longer cited.
- The statement that Perch and Bird-MAE pretraining data compositions (taxa) were not verified is kept as a hypothesis (P3).
