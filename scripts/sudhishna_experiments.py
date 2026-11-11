"""Create or refresh one Kaggle notebook folder per Sudhishna experiment.

The code lives in two master notebooks:
    notebooks/sudhishna_train/train.ipynb
    notebooks/sudhishna_infer/infer.ipynb
This script copies them into notebooks/sudhishna_train_<name>/ and sudhishna_infer_<name>/,
changing only the setting line at the top, and writes a private kernel-metadata.json for each.

Usage (from the repo root, after editing a master notebook):
    python3 scripts/sudhishna_experiments.py              # everything
    python3 scripts/sudhishna_experiments.py smoke        # only this experiment and its inference runs

Then run one on Kaggle:
    bash scripts/kaggle_run.sh notebooks/sudhishna_train_smoke
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_MASTER = ROOT / "notebooks/sudhishna_train/train.ipynb"
INFER_MASTER = ROOT / "notebooks/sudhishna_infer/infer.ipynb"
OWNER = "sudhishnajanavi"
JAMIE_MELS = "jmemei/sc4000-birdclef-m1-make-mels"
WRITEUP_MELS = f"{OWNER}/sc4000-birdclef-m2-make-mels-224"

# training run -> the notebook whose spectrograms it uses
TRAIN_RUNS = {
    "smoke": JAMIE_MELS,
    "sed20": JAMIE_MELS,
    "sed20-last": JAMIE_MELS,
    "sed20-hires": WRITEUP_MELS,
    "sed5": JAMIE_MELS,
}
# inference run -> (training run whose model it loads, seconds between chunks)
INFER_RUNS = {
    "smoke": ("smoke", 20),
    "sed20": ("sed20", 20),
    "sed20-overlap": ("sed20", 5),
    "sed20-last": ("sed20-last", 20),
    "sed20-last-overlap": ("sed20-last", 5),
    "sed20-hires": ("sed20-hires", 20),
    "sed20-hires-overlap": ("sed20-hires", 5),
    "sed5": ("sed5", 5),
}


def copy_notebook(master, setting, value, target):
    notebook = json.loads(master.read_text())
    hits = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source, n = re.subn(rf"^{setting} = \S+", f"{setting} = {value}", "".join(cell["source"]), flags=re.M)
        cell["source"] = source.splitlines(keepends=True)
        hits += n
    if hits != 1:
        sys.exit(f"expected exactly one {setting} line in {master}, found {hits}")
    target.write_text(json.dumps(notebook, indent=1) + "\n")


def write_metadata(folder, slug, code_file, gpu, internet, sources):
    metadata = {
        "id": f"{OWNER}/{slug}",
        "title": slug,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,        # course rule: code must not be public
        "enable_gpu": gpu,
        "enable_internet": internet,
        "dataset_sources": [],
        "competition_sources": ["birdclef-2025"],
        "kernel_sources": sources,
    }
    if gpu:
        metadata["machine_shape"] = "NvidiaTeslaT4"
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main(names):
    known = set(TRAIN_RUNS) | set(INFER_RUNS)
    unknown = set(names) - known
    if unknown:
        sys.exit(f"unknown experiment(s): {sorted(unknown)}; choose from {sorted(known)}")

    for name, mels_kernel in TRAIN_RUNS.items():
        if names and name not in names:
            continue
        folder = ROOT / f"notebooks/sudhishna_train_{name}"
        folder.mkdir(exist_ok=True)
        copy_notebook(TRAIN_MASTER, "EXPERIMENT", f'"{name}"', folder / "train.ipynb")
        # Internet downloads the ImageNet weights for the backbone.
        write_metadata(folder, f"sc4000-birdclef-m2-train-{name}", "train.ipynb", True, True, [mels_kernel])
        print(f"wrote {folder.relative_to(ROOT)}")

    for name, (train_run, stride) in INFER_RUNS.items():
        if names and name not in names:
            continue
        folder = ROOT / f"notebooks/sudhishna_infer_{name}"
        folder.mkdir(exist_ok=True)
        copy_notebook(INFER_MASTER, "STRIDE_SEC", str(stride), folder / "infer.ipynb")
        # Competition rules: CPU only, internet off.
        write_metadata(folder, f"sc4000-birdclef-m2-infer-{name}", "infer.ipynb", False, False,
                       [f"{OWNER}/sc4000-birdclef-m2-train-{train_run}"])
        print(f"wrote {folder.relative_to(ROOT)} (model from {train_run}, stride {stride} s)")


if __name__ == "__main__":
    main(sys.argv[1:])
