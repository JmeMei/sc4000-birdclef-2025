"""Create/refresh one Kaggle notebook folder per Member 1 ablation experiment.

The training code lives in ONE master notebook: notebooks/member1_train/train.ipynb.
This script copies it into notebooks/member1_train_<experiment>/, changing only the
`EXPERIMENT = "..."` line, and writes a private GPU kernel-metadata.json for each.
It also writes notebooks/member1_infer_<experiment>/: the baseline inference notebook
(unchanged) set up to load that experiment's trained model, CPU-only and offline.

Usage (from the repo root, after editing the master notebook):
    python3 scripts/m1_experiments.py              # all experiments
    python3 scripts/m1_experiments.py crop focal   # only these

Then run one on Kaggle with:
    bash scripts/kaggle_run.sh notebooks/member1_train_crop
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "notebooks/member1_train/train.ipynb"
INFER_MASTER = ROOT / "notebooks/infer_baseline/infer_baseline.ipynb"
EXPERIMENTS = ["base", "crop", "noise", "seclabel", "focal",
               "v2", "v2-nocrop", "v2-nonoise", "v2-nosec", "v2-focal"]
MELS_KERNEL = "jmemei/sc4000-birdclef-m1-make-mels"


def main(names):
    unknown = set(names) - set(EXPERIMENTS)
    if unknown:
        sys.exit(f"unknown experiment(s): {sorted(unknown)}; choose from {EXPERIMENTS}")
    nb = json.loads(MASTER.read_text())

    for name in names or EXPERIMENTS:
        copy = json.loads(json.dumps(nb))
        hits = 0
        for cell in copy["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            new, n = re.subn(r'^EXPERIMENT = "[^"]*"', f'EXPERIMENT = "{name}"', src, flags=re.M)
            cell["source"], hits = new, hits + n
        if hits != 1:
            sys.exit(f"expected exactly one EXPERIMENT line in {MASTER}, found {hits}")

        slug = f"sc4000-birdclef-m1-train-{name}"
        folder = ROOT / f"notebooks/member1_train_{name}"
        folder.mkdir(exist_ok=True)
        (folder / "train.ipynb").write_text(json.dumps(copy, indent=1))
        (folder / "kernel-metadata.json").write_text(json.dumps({
            "id": f"jmemei/{slug}",
            "title": slug,
            "code_file": "train.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,          # course rule: code must not be public
            "enable_gpu": True,
            "machine_shape": "NvidiaTeslaT4",
            "enable_internet": True,     # downloads ImageNet weights for timm
            "dataset_sources": [],
            "competition_sources": ["birdclef-2025"],
            "kernel_sources": [MELS_KERNEL],
        }, indent=2) + "\n")
        print(f"wrote {folder.relative_to(ROOT)}  (EXPERIMENT = {name!r})")

        # Matching CPU/offline inference kernel: the baseline inference notebook, unchanged,
        # reading model.pth / cfg.json / species.json from this experiment's training kernel.
        infer_slug = f"sc4000-birdclef-m1-infer-{name}"
        infer_dir = ROOT / f"notebooks/member1_infer_{name}"
        infer_dir.mkdir(exist_ok=True)
        (infer_dir / "infer.ipynb").write_text(INFER_MASTER.read_text())
        (infer_dir / "kernel-metadata.json").write_text(json.dumps({
            "id": f"jmemei/{infer_slug}",
            "title": infer_slug,
            "code_file": "infer.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,         # competition rule: CPU only
            "enable_internet": False,    # competition rule: internet off
            "dataset_sources": [],
            "competition_sources": ["birdclef-2025"],
            "kernel_sources": [f"jmemei/{slug}"],
        }, indent=2) + "\n")
        print(f"wrote {infer_dir.relative_to(ROOT)}  (model from jmemei/{slug})")


if __name__ == "__main__":
    main(sys.argv[1:])
