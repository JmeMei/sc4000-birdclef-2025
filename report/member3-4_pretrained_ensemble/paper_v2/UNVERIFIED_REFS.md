# Unverified references

Source: `notes/R5_citations.md` (checked 2026-11-08).

- Pseudo-Label (Lee 2013): omitted. No authoritative page was found, so it is not cited.
- MAE (He et al.): cited as arXiv only (`he2021masked`, year from the arXiv id). The CVPR 2022 venue and pages were not verified (the CVF page returned 403).
- Stochastic depth (Huang et al.): cited as arXiv only (`huang2016deep`, year from the arXiv id). ECCV 2016 and pages were not verified.
- Mixup (`zhang2018mixup`), focal loss (`lin2017focal`), Audio-MAE (`huang2022masked`) and Noisy Student (`xie2020selftraining`): venue and year come from the arXiv listing or the arXiv id only. Official proceedings pages (OpenReview, CVF, NeurIPS) were blocked or not opened. For mixup and focal loss the BibTeX entry is an arXiv `@misc` with year 2017 taken from the id; the mixup key still reads 2018 because the key is kept as in the notes.
- Ghani et al. (`ghani2023global`) and BirdNET (`kahl2021birdnet`): verified through the Crossref record and arXiv, not the publisher page.
- Sydorskyi and Goncalves (`sydorskyi2025tackling`): the 0.928 private / 0.925 public figures and the description of the system (strong baseline, in-domain transfer learning, semi-supervised distillation, postprocessing, ensembling) come from a non-CEUR listing (publications.kse.ua), because the CEUR PDF text was not machine-readable. The paper PDF itself was not read.
- Bird-MAE (`rauch2025can`): cited as arXiv with "Accepted at TMLR" noted; year, volume and pages of the journal version were not seen. The probing figures quoted in the paper (up to 37 percentage points mAP) come from the arXiv abstract, and the page read was truncated.
- BirdSet (`rauch2025birdset`): ICLR 2025 spotlight is from the arXiv comment; the OpenReview page was not opened.

## Added for paper_v2

- Bird-MAE novelty statement: based on `notes/R6_birdmae_novelty.md`. Full text of the four 2025 BirdCLEF+ working notes and of 20 of 21 BirdCLEF+ 2026 notes could not be read; Kaggle notebooks and discussions were not searchable. The paper states the search as limited and does not use the word "first".
- Bird-MAE pretraining data (Xeno-canto snapshot via BirdSet) and the claim that BirdSet evaluation sets are bird soundscapes come from the arXiv page as read in R6 (first 100k of 120k characters); taxonomic composition was not explicitly confirmed.
- Perch 2.0 training data (Xeno-canto and iNaturalist) is taken from the project notes (`report/member3-4_pretrained_ensemble/IMPLEMENTATION_GUIDE.md`, `STATE.md`); the Perch 2.0 paper text was not re-read for this paper, and its taxonomic composition was not verified.
- The 2026 paper that mentions Bird-MAE (arXiv 2607.14474) is described in the text but not cited, because it is not in `references.bib`.
