# R4 Kaggle CLI (docs-based; `--help` NOT run, no shell was available during this lookup)

Sources: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md , .../docs/kernels.md , .../docs/competitions.md ; P100 name from https://github.com/Kaggle/kaggle-cli/pull/916.diff (unmerged PR) and older docs.

## (a) kernel-metadata.json
Fields: id (user/slug), id_no, title, code_file, language ("python"), kernel_type ("script"|"notebook"), is_private (default true), enable_gpu (default false), enable_internet (default false), machine_shape (accelerator string), dataset_sources ["user/dataset"], competition_sources ["comp-slug"], kernel_sources ["user/kernel"], model_sources ["username/model-slug/framework/variation-slug/version-number"].
machine_shape values seen: NvidiaTeslaT4 (docs: "T4 x2", default GPU), NvidiaL4, NvidiaL4X1, NvidiaTeslaA100, NvidiaH100, NvidiaRtxPro6000, TpuV5E8, TpuV6E8; "" in sample. Older docs/PR also mention NvidiaTeslaP100 (PR calls it the "default GPU" - conflicts with main docs saying T4). Some are restricted to specific competitions/admins.
Push flag: `kaggle kernels push -p <dir> [--accelerator <ID>] [-t <seconds>] [--no-run]`. So: T4 = `--accelerator NvidiaTeslaT4` (or machine_shape); P100 = `NvidiaTeslaP100` (unverified in 2.2.5).
Pull: `kaggle kernels pull <owner/slug[/ver]> -p dir -m` generates metadata.

## (b) Code-competition submit
`kaggle competitions submit <COMP> -k <owner/kernel-slug> -v <version> -f <output file> -m "<msg>"`
(-k "kernel to submit (for code competitions)", -v kernel version e.g. 2, -f file, -m message.) UNVERIFIED: the exact -f value (presumably submission.csv, the file the notebook writes to /kaggle/working), whether late submissions work this way, and any internet-off requirement (docs silent; the competition rules require internet off for the notebook).

## (c) Status / output / scores
- `kaggle kernels status <owner/slug[/ver]>`
- `kaggle kernels output <owner/slug[/ver]> -p <dir> [-o] [--file-pattern REGEX]`
- `kaggle competitions submissions <COMP> [-v(csv)] [-q]` (columns undocumented)
- `kaggle competitions submission <SUBMISSION_REF>` shows Status, Public Score, Private Score (blank in doc sample), Description, Date, Error.
Note: `-v` means CSV for `submissions` but kernel version for `submit`. Whether private score is populated for late submissions via CLI: UNVERIFIED (may need web UI).
