# R6: Bird-MAE novelty check (web lookup, 2026-11-08)

## What was searched / seen
1. Web search "Bird-MAE BirdCLEF+ 2025": no paper or write-up evaluating Bird-MAE on BirdCLEF+ 2025. Surfaced only DS@GT 2026 paper (see below).
2. CEUR Vol-4038 index (https://ceur-ws.org/Vol-4038/): 4 BirdCLEF+ papers: 232 (overview), 249 (Miyaguchi et al., token distillation), 254 (Gokulnath et al., one detector per bird), 256 (Sydorskyi & Goncalves, 2nd place).
   Full text of all 4 PDFs: NOT readable (the web fetch returned compressed PDF streams). 0 of 4 checked in full text. Abstract-level titles show no Bird-MAE. Unresolved.
3. CEUR Vol-4283 index (https://ceur-ws.org/Vol-4283/): CLEF 2026 working notes, BirdCLEF+ 2026 = Pantanal (234 species), 21 papers listed (paper317..370; the last ~4000 chars of the index page were not read, so possibly a few more).
   - paper353 (Miyaguchi, Gustineli, Cheung, "Can Tokens Compete?"): read via HF/arXiv mirror https://huggingface.co/papers/2607.14474.md (arXiv 2607.14474). READ. Mentions Bird-MAE in related work ("Bird-MAE adapts masked autoencoding to birds"), discussion 6.2 (transformer scaling vs Perch; Bird-MAE only had <=10 s context), and Appendix E (linear probe known-bad for MAE features, 13 vs 49 mAP gap; cites Bird-MAE ViT-L prototypical probe cmAP 0.55 on BirdSet). They did NOT run Bird-MAE; encoders evaluated: Perch (0.866 public), BirdNET (0.814), BEATs (0.777), WavTokenizer etc.
   - paper321, 322, 335, 363: PDF fetch returned compressed streams, unreadable.
   - Remaining 16 papers in the 2026 list: not attempted (PDFs unreadable by the tool).
   - Tally: 2026 papers read = 1 of 21; unreadable/attempted = 4; not attempted = 16. 2025 papers read = 0 of 4.
4. Secondary search ("Bird-MAE" OR "BirdMAE" BirdCLEF 2026): only hits were paper353, emergentmind BirdMAE topic page, and an unrelated iNatSounds MAE paper (arXiv 2605.14031, Liu et al., FGVC@CVPR 2026; evaluates on iNatSounds, no BirdCLEF/Bird-MAE evidence seen on abstract page). Colling's BirdCLEF+ 2026 page (gillescolling.com) has no Bird-MAE mention.
5. Kaggle search (allowed domains kaggle/github/arxiv) for BirdCLEF+ 2025 + Bird-MAE: no Kaggle write-up or notebook mentioning Bird-MAE found. Kaggle pages not individually opened; Kaggle notebook full-text search is not available through this tool, so notebooks are NOT covered.
6. Bird-MAE paper itself (https://arxiv.org/html/2504.12880, first 100k of 120k chars read): downstream = BirdSet 8 tasks: HSN (validation/dev), POW, PER, NES, UHH, NBP, SSW, SNE. No BirdCLEF mention. Pretrain on XCL (Xeno-canto, BirdSet). 5 s segments, 32 kHz. ViT-B/16, L/16, H/16. Version seen: v4. Authors: Rauch, Heinrich, Moummad, Joly, Sick, Scholz. None of the 8 is BirdCLEF+ 2025 or multi-taxa (birds only, as far as the page shows; taxa not explicitly confirmed).

## Findings
- Direct prior use of Bird-MAE on BirdCLEF+ 2025 or 2026 data: none found.
- Indirect: paper353 cites Bird-MAE (BirdSet numbers only) but did not evaluate it.
- Paper cites to consider: Rauch et al. 2504.12880 (must cite); optionally Miyaguchi et al. 2607.14474 (BirdCLEF+ 2026 token paper).

## Not established
- Full text of 4/4 BirdCLEF+ 2025 papers and 20/21 of 2026 papers; Kaggle notebooks and discussion bodies; Google Scholar "cited by" list for 2504.12880 (not accessible); tail of 2026 index; last 20k chars of Bird-MAE html.

## Suggested wording
"To our knowledge, and based on a limited search of the CLEF 2025/2026 BirdCLEF working notes, Kaggle write-ups and arXiv, we did not find a prior report of frozen Bird-MAE-Base features evaluated on BirdCLEF+ 2025 multi-taxa soundscapes." Do not write "first".

## BibTeX (fields seen on arXiv page; year from arXiv ID 2504 = April 2025)
@misc{rauch2025birdmae,
  title         = {Can Masked Autoencoders Also Listen to Birds?},
  author        = {Rauch, Lukas and Heinrich, Ren{\'e} and Moummad, Ilyass and Joly, Alexis and Sick, Bernhard and Scholz, Christoph},
  year          = {2025},
  eprint        = {2504.12880},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2504.12880}
}
(Year/primary class: categories cs.LG, cs.SD, eess.AS seen in PDF; year inferred from ID.)

## Sources
https://ceur-ws.org/Vol-4038/ ; https://ceur-ws.org/Vol-4283/ ; https://huggingface.co/papers/2607.14474.md ; https://arxiv.org/abs/2504.12880 ; https://arxiv.org/html/2504.12880 ; https://arxiv.org/abs/2605.14031 ; https://gillescolling.com/side-projects/bird-clef-2026/
