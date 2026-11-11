# R5 citations (checked 2026-11-08)

## A) AAAI Author Kit (aaai.org only)
- Current kit: AAAI-27. Page https://aaai.org/conference/aaai/aaai-27/submission-instructions/ links to https://aaai.org/authorkit27/ . Fetching that URL returned an application/zip (5.2 MB), so it IS the direct download.
- Zip contents seen (folder AuthorKit27/): aaai2027.sty, aaai2027.bst, aaai2027.bib, CameraReady2027.tex, AnonymousSubmission2027.tex, ReproducibilityChecklist.tex, Figures/, Word/ (docx templates). Class/style: aaai2027.sty; bibliography style: aaai2027.bst.
- Not seen: whether an AAAI-26 kit link still exists (AAAI Press page proceedings.aaai.org/info still mentions the 2026 kit). Not checked because 27 is newer.

## B) References

### VERIFIED
1. Overview of BirdCLEF+ 2025 (CEUR Vol-4038 paper 232; pages 2909-2919, NOT p.232). https://ceur-ws.org/Vol-4038/ , PDF https://ceur-ws.org/Vol-4038/paper_232.pdf
Authors: Juan Sebastian Canas, Stefan Kahl, Tom Denton, Maria Paula Toro-Gomez, Susana Rodriguez-Buritica, Jose Luis Benavides-Lopez, Juan Sebastian Ulloa, Paula Caycedo-Rosales, Holger Klinck, Herve Goeau, Willem-Pier Vellinga, Robert Planque, Alexis Joly. Volume: Working Notes of CLEF 2025, eds. Faggioli, Ferro, Rosso, Spina; Madrid 9-12 Sep 2025. (Accents were stripped in my notes; the page has them.)
```
@inproceedings{canas2025birdclef,
  author = {Juan Sebasti{\'a}n Ca{\~n}as and Stefan Kahl and Tom Denton and Mar{\'i}a Paula Toro-G{\'o}mez and Susana Rodriguez-Buritica and Jose Luis Benavides-Lopez and Juan Sebasti{\'a}n Ulloa and Paula Caycedo-Rosales and Holger Klinck and Herv{\'e} Go{\"e}au and Willem-Pier Vellinga and Robert Planqu{\'e} and Alexis Joly},
  title = {Overview of {BirdCLEF+} 2025: Multi-Taxonomic Sound Identification in the Middle Magdalena, Colombia},
  booktitle = {Working Notes of the Conference and Labs of the Evaluation Forum (CLEF 2025)},
  series = {CEUR Workshop Proceedings},
  volume = {4038},
  pages = {2909--2919},
  year = {2025},
  url = {https://ceur-ws.org/Vol-4038/paper_232.pdf}
}
```
2. Sydorskyi & Goncalves, paper 256, pages 3208-3228. https://ceur-ws.org/Vol-4038/paper_256.pdf (listing page seen; PDF text not machine-readable for me). Title: Tackling Domain Shift in Bird Audio Classification via Transfer Learning and Semi-Supervised Distillation: A Case Study on BirdCLEF+ 2025.
Citable sentence (from KSE listing https://publications.kse.ua/publications/tackling-domain-shift-bird-audio-1909, not the CEUR PDF): the 2nd-place system combines a strong baseline, in-domain transfer learning, semi-supervised distillation against domain shift, postprocessing and ensembling, scoring ROC-AUC 0.928 private / 0.925 public.
```
@inproceedings{sydorskyi2025tackling,
  author = {Volodymyr Sydorskyi and Fernando Gon{\c{c}}alves},
  title = {Tackling Domain Shift in Bird Audio Classification via Transfer Learning and Semi-Supervised Distillation: A Case Study on {BirdCLEF+} 2025},
  booktitle = {Working Notes of the Conference and Labs of the Evaluation Forum (CLEF 2025)},
  series = {CEUR Workshop Proceedings},
  volume = {4038},
  pages = {3208--3228},
  year = {2025},
  url = {https://ceur-ws.org/Vol-4038/paper_256.pdf}
}
```
3. DS@GT, arXiv 2507.08236 (https://arxiv.org/abs/2507.08236). Title: Distilling Spectrograms into Tokens: Fast and Lightweight Bioacoustic Classification for BirdCLEF+ 2025. Authors: Anthony Miyaguchi, Murilo Gustineli, Adrian Cheung. Submitted 11 Jul 2025; "Working note submitted to CLEF 2025 under the LifeCLEF lab". Note: the title does not contain "DS@GT".
```
@misc{miyaguchi2025distilling,
  author = {Anthony Miyaguchi and Murilo Gustineli and Adrian Cheung},
  title = {Distilling Spectrograms into Tokens: Fast and Lightweight Bioacoustic Classification for {BirdCLEF+} 2025},
  year = {2025},
  eprint = {2507.08236},
  archivePrefix = {arXiv}
}
```
4. BirdCLEF+ 2026 notes, CEUR Vol-4283 "CLEF 2026 Working Notes" (eds. Eva Sanchez Salido, Alberto Barron-Cedeno, Alba Garcia Seco de Herrera, Sean MacAvaney, Julia Maria Struss). Both exist:
 - paper 323: Rashmi Banthia, Stefano Morelli, Aman Kapoor, "Listening to the Pantanal: Runtime-Constrained Distillation and Diversity-Aware Ensembling for BirdCLEF+ 2026", pp. 4386-4405. https://ceur-ws.org/Vol-4283/paper323.pdf
 - paper 330: Yannan Chen, "Class Interaction as Domain Recalibration: A Variance-Aware Sound-Event-Detection Pipeline for BirdCLEF+ 2026", pp. 4498-4512. https://ceur-ws.org/Vol-4283/paper330.pdf
 Both are participant papers, not an overview. (Page read was truncated at 100000 of 104121 chars; the unread tail was not needed.)
```
@inproceedings{banthia2026pantanal,
  author = {Rashmi Banthia and Stefano Morelli and Aman Kapoor},
  title = {Listening to the Pantanal: Runtime-Constrained Distillation and Diversity-Aware Ensembling for {BirdCLEF+} 2026},
  booktitle = {CLEF 2026 Working Notes},
  series = {CEUR Workshop Proceedings},
  volume = {4283},
  pages = {4386--4405},
  year = {2026},
  url = {https://ceur-ws.org/Vol-4283/paper323.pdf}
}
@inproceedings{chen2026class,
  author = {Yannan Chen},
  title = {Class Interaction as Domain Recalibration: A Variance-Aware Sound-Event-Detection Pipeline for {BirdCLEF+} 2026},
  booktitle = {CLEF 2026 Working Notes},
  series = {CEUR Workshop Proceedings},
  volume = {4283},
  pages = {4498--4512},
  year = {2026},
  url = {https://ceur-ws.org/Vol-4283/paper330.pdf}
}
```
5. Perch 2.0, https://arxiv.org/abs/2508.04665. Perch 2.0: The Bittern Lesson for Bioacoustics. Bart van Merrienboer, Vincent Dumoulin, Jenny Hamer, Lauren Harrell, Andrea Burns, Tom Denton. v1 6 Aug 2025, v2 5 Jan 2026. No venue on page.
```
@misc{vanmerrienboer2025perch,
  author = {Bart van Merri{\"e}nboer and Vincent Dumoulin and Jenny Hamer and Lauren Harrell and Andrea Burns and Tom Denton},
  title = {Perch 2.0: The Bittern Lesson for Bioacoustics},
  year = {2025},
  eprint = {2508.04665},
  archivePrefix = {arXiv}
}
```
6. Ghani et al. Verified via Crossref record (DOI registry) https://api.crossref.org/works/10.1038/s41598-023-49989-z and arXiv https://arxiv.org/abs/2307.06292. Nature page itself was blocked (redirect), so publisher page NOT seen. Scientific Reports 13, article 22876, 2023, DOI 10.1038/s41598-023-49989-z. Authors: Burooj Ghani, Tom Denton, Stefan Kahl, Holger Klinck.
```
@article{ghani2023global,
  author = {Burooj Ghani and Tom Denton and Stefan Kahl and Holger Klinck},
  title = {Global birdsong embeddings enable superior transfer learning for bioacoustic classification},
  journal = {Scientific Reports},
  volume = {13},
  number = {22876},
  year = {2023},
  doi = {10.1038/s41598-023-49989-z}
}
```
7. Bird-MAE, https://arxiv.org/abs/2504.12880. Can Masked Autoencoders Also Listen to Birds? Lukas Rauch, Rene Heinrich, Ilyass Moummad, Alexis Joly, Bernhard Sick, Christoph Scholz. v1 17 Apr 2025, v4 19 Aug 2025; page says "Accepted at TMLR" (year/volume/pages not seen).
Citable: Bird-MAE sets new SOTA on BirdSet's multi-label benchmark. Probing (from arXiv HTML): abstract "Bird-MAE's prototypical probes outperform linear probing by up to 37 percentage points in mean average precision"; section 6.1: prototypical probing "closes the gap to fine-tuning to approximately 3 pp mAP on average". Table 7 (ViT-L/16, mAP %): HSN val linear 12.44 vs prototypical 49.97; POW linear 16.20 vs prototypical 51.73. These are ViT-L numbers; Base numbers not seen. Metric = mAP, not AUROC. (Page read truncated at 100000 of 120396 chars.)
```
@misc{rauch2025can,
  author = {Lukas Rauch and Ren{\'e} Heinrich and Ilyass Moummad and Alexis Joly and Bernhard Sick and Christoph Scholz},
  title = {Can Masked Autoencoders Also Listen to Birds?},
  year = {2025},
  eprint = {2504.12880},
  archivePrefix = {arXiv},
  note = {Accepted at TMLR}
}
```
8. BirdSet, https://arxiv.org/abs/2403.10380. BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics. Lukas Rauch, Raphael Schwinger, Moritz Wirth, Rene Heinrich, Denis Huseljic, Marek Herde, Jonas Lange, Stefan Kahl, Bernhard Sick, Sven Tomforde, Christoph Scholz. v1 15 Mar 2024, v6 18 May 2025; comment "accepted as spotlight @ICLR2025" (OpenReview page not opened; no pages).
```
@inproceedings{rauch2025birdset,
  author = {Lukas Rauch and Raphael Schwinger and Moritz Wirth and Ren{\'e} Heinrich and Denis Huseljic and Marek Herde and Jonas Lange and Stefan Kahl and Bernhard Sick and Sven Tomforde and Christoph Scholz},
  title = {{BirdSet}: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2025},
  note = {Spotlight. arXiv:2403.10380}
}
```
9. Foundation Models for Bioacoustics -- a Comparative Review, https://arxiv.org/abs/2508.01277. Raphael Schwinger, Paria Vali Zadeh, Lukas Rauch, Mats Kurz, Tom Hauschild, Sam Lapp, Sven Tomforde. v1 2 Aug 2025, v2 29 Mar 2026. "Preprint", no venue.
```
@misc{schwinger2025foundation,
  author = {Raphael Schwinger and Paria Vali Zadeh and Lukas Rauch and Mats Kurz and Tom Hauschild and Sam Lapp and Sven Tomforde},
  title = {Foundation Models for Bioacoustics -- a Comparative Review},
  year = {2025},
  eprint = {2508.01277},
  archivePrefix = {arXiv}
}
```
11. Audio-MAE, https://arxiv.org/abs/2207.06405. Masked Autoencoders that Listen. Po-Yao Huang, Hu Xu, Juncheng Li, Alexei Baevski, Michael Auli, Wojciech Galuba, Florian Metze, Christoph Feichtenhofer. "Accepted at NeurIPS 2022". (Proceedings page not reachable; no pages/volume.)
```
@inproceedings{huang2022masked,
  author = {Po-Yao Huang and Hu Xu and Juncheng Li and Alexei Baevski and Michael Auli and Wojciech Galuba and Florian Metze and Christoph Feichtenhofer},
  title = {Masked Autoencoders that Listen},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2022},
  note = {arXiv:2207.06405}
}
```
12. EfficientNet, PMLR https://proceedings.mlr.press/v97/tan19a.html : Mingxing Tan, Quoc Le; ICML 2019, PMLR 97, pp. 6105-6114.
```
@inproceedings{tan2019efficientnet,
  author = {Mingxing Tan and Quoc Le},
  title = {{EfficientNet}: Rethinking Model Scaling for Convolutional Neural Networks},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  volume = {97},
  pages = {6105--6114},
  year = {2019}
}
```
13. EfficientNetV2, https://proceedings.mlr.press/v139/tan21a.html : ICML 2021, PMLR 139, pp. 10096-10106.
```
@inproceedings{tan2021efficientnetv2,
  author = {Mingxing Tan and Quoc Le},
  title = {{EfficientNetV2}: Smaller Models and Faster Training},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  volume = {139},
  pages = {10096--10106},
  year = {2021}
}
```
14. mixup, https://arxiv.org/abs/1710.09412 : mixup: Beyond Empirical Risk Minimization; Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz. Listing says "ICLR camera-ready version"; year and OpenReview page NOT seen (OpenReview blocked).
```
@inproceedings{zhang2018mixup,
  author = {Hongyi Zhang and Moustapha Ciss{\'e} and Yann N. Dauphin and David Lopez-Paz},
  title = {mixup: Beyond Empirical Risk Minimization},
  booktitle = {International Conference on Learning Representations (ICLR)},
  note = {arXiv:1710.09412}
}
```
15. Focal loss, https://arxiv.org/abs/1708.02002 : Focal Loss for Dense Object Detection; Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar. arXiv page names no venue (ICCV 2017 not seen on a page; CVF page not opened).
Prior-bias init CONFIRMED (arXiv HTML https://arxiv.org/html/1708.02002, section "Initialization"), verbatim opening: "For the final conv layer of the classification subnet, we set the bias initialization to b = -log((1-pi)/pi)," (tool could only return the first 125 chars of the sentence; it paraphrased the rest as pi being the foreground confidence every anchor has at the start of training). Also: "We use pi=.01 in all experiments, although results are robust to the exact value."
```
@misc{lin2017focal,
  author = {Tsung-Yi Lin and Priya Goyal and Ross Girshick and Kaiming He and Piotr Doll{\'a}r},
  title = {Focal Loss for Dense Object Detection},
  eprint = {1708.02002},
  archivePrefix = {arXiv}
}
```
16. SpecAugment, https://arxiv.org/abs/1904.08779 : Daniel S. Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D. Cubuk, Quoc V. Le. Proc. Interspeech 2019, pp. 2613-2617, DOI 10.21437/Interspeech.2019-2680.
```
@inproceedings{park2019specaugment,
  author = {Daniel S. Park and William Chan and Yu Zhang and Chung-Cheng Chiu and Barret Zoph and Ekin D. Cubuk and Quoc V. Le},
  title = {{SpecAugment}: A Simple Data Augmentation Method for Automatic Speech Recognition},
  booktitle = {Proc. Interspeech 2019},
  pages = {2613--2617},
  doi = {10.21437/Interspeech.2019-2680}
}
```
18. Noisy Student, https://arxiv.org/abs/1911.04252 : Self-training with Noisy Student improves ImageNet classification; Qizhe Xie, Minh-Thang Luong, Eduard Hovy, Quoc V. Le; listing says "CVPR 2020" (pages not seen).
```
@inproceedings{xie2020selftraining,
  author = {Qizhe Xie and Minh-Thang Luong and Eduard Hovy and Quoc V. Le},
  title = {Self-training with Noisy Student improves {ImageNet} classification},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020},
  note = {arXiv:1911.04252}
}
```
20. Hinton, https://arxiv.org/abs/1503.02531 : Distilling the Knowledge in a Neural Network; Geoffrey Hinton, Oriol Vinyals, Jeff Dean; submitted 9 Mar 2015; "Presented at the NIPS 2014 Deep Learning Workshop".
```
@misc{hinton2015distilling,
  author = {Geoffrey Hinton and Oriol Vinyals and Jeff Dean},
  title = {Distilling the Knowledge in a Neural Network},
  year = {2015},
  eprint = {1503.02531},
  archivePrefix = {arXiv},
  note = {NIPS 2014 Deep Learning Workshop}
}
```
21. BirdNET, Crossref record https://api.crossref.org/works/10.1016/j.ecoinf.2021.101236 (publisher page not opened): BirdNET: A deep learning solution for avian diversity monitoring; Stefan Kahl, Connor M. Wood, Maximilian Eibl, Holger Klinck; Ecological Informatics 61, article 101236, 2021, DOI 10.1016/j.ecoinf.2021.101236.
```
@article{kahl2021birdnet,
  author = {Stefan Kahl and Connor M. Wood and Maximilian Eibl and Holger Klinck},
  title = {{BirdNET}: A deep learning solution for avian diversity monitoring},
  journal = {Ecological Informatics},
  volume = {61},
  number = {101236},
  year = {2021},
  doi = {10.1016/j.ecoinf.2021.101236}
}
```

### PARTLY VERIFIED / UNVERIFIED
10. MAE (He et al.): arXiv page seen (https://arxiv.org/abs/2111.06377: Masked Autoencoders Are Scalable Vision Learners; Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, Ross Girshick; arXiv lists no venue, "Tech report"). The CVF page returned 403, so the CVPR 2022 venue is UNVERIFIED by page. A web search snippet (secondary) gave CVPR 2022 pp. 16000-16009, DOI 10.1109/CVPR52688.2022.01553; other sites list 15979-15988. Verify at https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html before citing pages.
17. Stochastic depth (Huang, Sun, Liu, Sedra, Weinberger): arXiv 1603.09382 page seen ("Deep Networks with Stochastic Depth"; v1 30 Mar 2016) but lists NO venue; ECCV 2016 and pages UNVERIFIED (check Springer, ECCV 2016 LNCS).
19. Pseudo-Label (Lee 2013): no authoritative page opened. Search returned only secondary sources, one of which says ICLR workshop; ICML 2013 Workshop on Challenges in Representation Learning is from memory only. UNVERIFIED; check the ICML 2013 workshop PDF.

### Gaps noted
- Items 14, 15, 17, 18 and 11: venue is on arXiv comment or not shown; official proceedings pages (OpenReview, CVF, NeurIPS, Springer) were blocked or unopened.
- Item 6/21: verified via Crossref, not the publisher page.
- Item 2: abstract text not read from CEUR PDF itself (PDF streams unreadable).
