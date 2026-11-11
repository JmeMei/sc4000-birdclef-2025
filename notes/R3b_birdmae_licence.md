# Bird-MAE licence check (2026-11-08)

Method: summaries of the pages, not the raw pages. Treat as unverified until the raw pages are checked.

| Source | URL | Finding |
|---|---|---|
| HF model card Bird-MAE-Base | https://huggingface.co/DBD-research-group/Bird-MAE-Base | No licence field or section shown. Training dataset listed: DBD-research-group/BirdSet. |
| GitHub Bird-MAE | https://github.com/DBD-research-group/Bird-MAE | No LICENSE file, no licence in sidebar, no README licence section. |
| GitHub BirdSet | https://github.com/DBD-research-group/BirdSet | Sidebar: "BSD-3-Clause license". Covers the BirdSet code only. README describes XCL as "Complete Xeno-Canto snapshot with focals for large (pre-) training"; no Xeno-canto or XCL licence terms stated. |
| arXiv 2504.12880 | https://arxiv.org/abs/2504.12880 | Only the arXiv non-exclusive distribution licence for the paper text. Nothing for code, weights or data. |

## Conclusion
- Bird-MAE weights: no licence declared in the HF card (as fetched).
- Bird-MAE code repo: no licence declared.
- BirdSet code: BSD-3-Clause. Not stated to apply to Bird-MAE weights.
- Dataset terms: not stated in the sources checked. Xeno-canto recordings carry per-recording Creative Commons licences (many CC BY-NC-SA). This is general knowledge and was not verified here. It could affect commercial reuse, but this is a course project.

## Unverified
- The HF card's YAML metadata and the repo's file tree were seen only through summaries.
- The BirdSet HF dataset card (https://huggingface.co/datasets/DBD-research-group/BirdSet) was not checked. It is the likely place for data licence terms.
- The paper PDF full text was not read.
