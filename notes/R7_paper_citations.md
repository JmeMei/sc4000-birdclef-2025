# R7 paper citations (2026-11-10). BibTeX in R7_references.bib

## A) AAAI-27 author kit
- Info page: https://aaai.org/conference/aaai/aaai-27/submission-instructions/ (links "author kit" to the URL below).
- Direct download (application/zip, 5.2 MB): https://aaai.org/authorkit27/
- Version: AAAI-27 (folder AuthorKit27/). Zip entries seen in the (truncated) listing: CameraReady2027.tex, aaai2027.bib, aaai2027.bst, Word/CameraReady2027.docx, Word/ReproducibilityChecklist.pdf. R5 also saw aaai2027.sty, AnonymousSubmission2027.tex, ReproducibilityChecklist.tex, Figures/. This session did not see .sty or AnonymousSubmission2027.tex in the listing (truncated).
- Submission rules (page above): AAAI two-column camera-ready style, US Letter, Type 1/TrueType fonts; up to 9 pages, of which 7 are content and pp. 8-9 references only; review version has no author/affiliation and no acknowledgements; camera-ready (main track) due 14 Dec 2026; source files only needed if accepted.
- Preamble: the page itself does not state it. From the project's copy D:\sc4000-kaggle\docs\paper\aaai2027.sty (not checked byte-for-byte against the downloaded zip): `\ProvidesPackage{aaai2027}[2027/10/21 ...]`; options `submission` (anonymous), `draft`, `preprint`, `ccby`; no option = camera-ready. Requires pdfLaTeX (guard against XeLaTeX/LuaLaTeX). Loads itself: iftex, fontenc T1, newtxtext, helvet, courier, placeins, xcolor.
- Anonymous: `\usepackage[submission]{aaai2027}`. Camera-ready: `\usepackage{aaai2027}`.
- Errors if loaded (forbidden): hyperref, bbm, authblk, balance, CJK, flushend, fullpage, geometry, navigator, savetrees, setspace, stfloats, tabu, titlesec, tocbibind, ulem, wrapfig. The project main.tex also lists float, indentfirst, layout, multicol, nameref as forbidden (comments from the kit template).
- Required in template: `\documentclass[letterpaper]{article}`, `\usepackage{aaai2027}`, `\usepackage[hyphens]{url}`, `\usepackage{graphicx}`, `\usepackage{natbib}` and `\usepackage{caption}` with no options. Also do not add times/helvet/courier.

## B) Verification
Format: item - source URL - status.
1. Overview BirdCLEF+ 2025, paper 232, pp. 2909-2919: https://ceur-ws.org/Vol-4038/ - verified. LifeCLEF 2025 overview is a separate item (Picek et al., LNCS 16089, pp. 338-362, DOI 10.1007/978-3-032-04354-2_19): seen only on HAL record https://hal.inrae.fr/hal-05408591v1 via search snippet; Springer not opened; full author list not seen. Vol-4038 has no single LifeCLEF overview (papers 231-235 are per-task). BirdCLEF+ overview is the right citation for the competition.
2. Paper 256, pp. 3208-3228: Vol-4038 index - verified.
3. DS@GT: Vol-4038 index gives paper 249, pp. 3111-3127, same title/authors as arXiv 2507.08236 (https://arxiv.org/abs/2507.08236) - verified.
4. CEUR Vol-4283 index https://ceur-ws.org/Vol-4283/ :
   - 330 (Yannan Chen, pp. 4498-4512): SED pipeline with raw-waveform MixUp, external in-domain data, semi-supervised noisy-student distillation, and a class-interaction recalibration module (search-engine summary of abstract/intro; PDF text unreadable by tool). No cross-family distillation evidence seen.
   - 340 (Musa Tur Farazi, Nufayer Jahan Reza, pp. 4663-4685): iterative semi-supervised distillation (pseudo-labels, noisy-student style); SED branch plus Perch-based ProtoSSM branch, per-class rank blending (search snippet). Cross-family/cross-model distillation UNCONFIRMED for both: PDFs are compressed streams the fetch tool cannot read. Read them manually at the URLs https://ceur-ws.org/Vol-4283/paper330.pdf and paper340.pdf.
   - Also seen: 323 (R5), 334 "Perch-Distilled ProtoSSM", 353 "Can Tokens Compete?" (Miyaguchi et al.), 317 Overview of BirdCLEF+ 2026 (Kahl et al., pp. 4315-4331).
5. Perch 2.0: https://arxiv.org/abs/2508.04665 - v1 6 Aug 2025 (after the 2025 competition closed, deadline 5 Jun 2025 per https://www.imageclef.org/BirdCLEF2025), v2 5 Jan 2026. No venue.
6. Bird-MAE: TMLR 2025, https://openreview.net/forum?id=GIBWR0Xo2J, bib from https://jmlr.org/tmlr/papers/bib/GIBWR0Xo2J.bib - verified (no volume/pages).
7. BirdSet: ICLR 2025 spotlight only per arXiv comment (https://arxiv.org/abs/2403.10380); OpenReview/ICLR page not opened.
8. MAE: Crossref https://api.crossref.org/works/10.1109/CVPR52688.2022.01553 : CVPR 2022, pp. 15979-15988. (CVF page 403; the 16000-16009 in R5 is contradicted by Crossref.)
9. Audio-MAE: NeurIPS 35 (2022) https://proceedings.neurips.cc/paper_files/paper/2022/hash/b89d5e209990b19e33b418e14f323998-Abstract-Conference.html - verified; no page numbers listed.
10. EfficientNet / V2: PMLR 97 and 139 (R5, unchanged).
11. AdaBN: arXiv 1603.04779 title "Revisiting Batch Normalization For Practical Domain Adaptation" (no venue on arXiv page; the ICLR 2017 workshop claim is UNVERIFIED). Journal: Pattern Recognition 80:109-117, 2018, DOI 10.1016/j.patcog.2018.03.005, titled "Adaptive Batch Normalization for practical domain adaptation" (Crossref). Cite the journal paper; note the title differs.
12. CORAL: Crossref https://api.crossref.org/works/10.1609/aaai.v30i1.10306 : AAAI 30(1), 2016. Pages not given.
13. Boudiaf et al.: https://proceedings.mlr.press/v202/boudiaf23a.html : ICML 2023, PMLR 202, pp. 2914-2931. Verified. (Whether it covers bioacoustics: not checked here; co-authors include Denton, van Merrienboer, Dumoulin.)
14. Tent: ICLR 2021 spotlight, https://iclr.cc/virtual/2021/poster/2874 - authors Wang, Shelhamer, Liu, Olshausen, Darrell. No pages (OpenReview).
15. Noisy Student: Crossref https://api.crossref.org/works/10.1109/CVPR42600.2020.01070 : CVPR 2020, pp. 10684-10695.
16. Pseudo-Label (Lee 2013): UNVERIFIED venue (search only returned secondary sources, one saying ICLR workshop). Check the ICML 2013 WREPL workshop page.
17. Hinton et al.: arXiv 1503.02531 (R5).
18. mixup: ICLR 2018 (secondary: https://mlanthology.org/iclr/2018/zhang2018iclr-mixup; OpenReview id r1Ddp1-Rb not opened).
19. Stochastic depth: Crossref https://api.crossref.org/works/10.1007/978-3-319-46493-0_39 : ECCV 2016, LNCS, pp. 646-661, DOI above. LNCS volume number not listed.
20. Focal loss: Crossref https://api.crossref.org/works/10.1109/ICCV.2017.324 : ICCV 2017, pp. 2999-3007. Prior-bias init confirmed in R5.
21. SpecAugment, 22. BirdNET: as R5.
23. Leaderboard scores:
   - DS@GT: best model-zoo model BirdSetEfficientNetB1, public 0.810, private 0.778. Stated in abstract and conclusion of arXiv 2507.08236 (https://arxiv.org/html/2507.08236v1, Table 11). Public 0.810 is higher than private 0.778. Search snippets give different numbers for paper 249 on CEUR (private 0.711 for the token approach) vs 0.520 in an arXiv abstract summary; the token-method score is unresolved, do not cite it.
   - 2nd place 0.928 private / 0.925 public: Sydorskyi & Goncalves, via https://publications.kse.ua/publications/tackling-domain-shift-bird-audio-1909 and search summary of paper 256; CEUR PDF text not read.
   - 1st place 0.930 private (0.933 public): only a search-engine summary of the overview (paper 232) and HAL record; PDF unreadable by tool (exceeds 10 MB fetch limit). The overview does not seem to name the winner in what we saw. Treat as UNVERIFIED at primary source.

## UNVERIFIED
- Contents of .sty/.tex inside the downloaded zip (not unzipped); kit .sty date vs the project copy's 2027/10/21.
- Papers 330/340 on cross-family distillation (PDF text unreadable).
- Lee 2013 venue; AdaBN ICLR workshop; BirdSet ICLR page; mixup/Tent OpenReview pages; Springer page for LifeCLEF overview; 1st-place 0.930 and 2nd-place 0.928 in the CEUR PDFs themselves.
