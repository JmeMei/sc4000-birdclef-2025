"""Build the three paper_v2 figures.

Real files read where possible:
  docs/_kaggle_logs/a3-probe-retry/probe_retry_results.json (latest version, v2)
  docs/_kaggle_logs/a3-mlp-head/head_cfg.json
Other values are copied from STATE.md (sweep table, probe-retry v1) and
reports/results_table.md (leaderboard, val AUC); see the constants below.
Grayscale only, vector PDF, Type 42 fonts, single-column size, no titles.
Optional argument: directory for 110-dpi PNG previews.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.2, "hatch.linewidth": 0.8,
})

HERE = Path(__file__).resolve().parent
LOGS = HERE.parents[2] / "docs" / "_kaggle_logs"
PREVIEW = Path(sys.argv[1]) if len(sys.argv) > 1 else None
SIZE = (3.3, 2.4)

# ---- real files -----------------------------------------------------------
retry = {r["name"]: r for r in json.load(
    open(LOGS / "a3-probe-retry" / "probe_retry_results.json"))["results"]}
mlp = json.load(open(LOGS / "a3-mlp-head" / "head_cfg.json"))["runs"]

lin_default = retry["linear_lr0.001"]["first"]          # 0.7498
lin_prior = retry["linear_lr0.001_priorbias"]["first"]  # 0.9626
mlp30_default = mlp["default"]["val_auc_first5s"]       # 0.9749
mlp30_prior = mlp["log_prior"]["val_auc_first5s"]       # 0.9707
# STATE.md, "A3 retry v1 results" (sweep, default bias): not in the v2 json
lin_lr3e3, lin_lr1e2, mlp12 = 0.8513, 0.9370, 0.9700

ARC = [  # label, value, hatch ('' default bias, '//' log-prior bias)
    ("Linear 1e-3, default", lin_default, ""),
    ("Linear 3e-3, default", lin_lr3e3, ""),
    ("Linear 1e-2, default", lin_lr1e2, ""),
    ("Linear 1e-3, log-prior", lin_prior, "///"),
    ("MLP 12 ep, default", mlp12, ""),
    ("MLP 30 ep, default", mlp30_default, ""),
    ("MLP 30 ep, log-prior", mlp30_prior, "///"),
]

# reports/results_table.md (val AUC first-5s, public, private)
LB = [("v2", 0.763, 0.789), ("A1", 0.82822, 0.81642),
      ("A2", 0.84774, 0.85226), ("A3", 0.77640, 0.78016)]
CUTOFF, GOAL = 0.776, 0.810
VAL = {"A1": 0.9832, "A2": 0.9916, "A3": 0.9749}


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight")
    if PREVIEW:
        PREVIEW.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW / f"{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def arc():
    fig, ax = plt.subplots(figsize=SIZE)
    y = list(range(len(ARC)))[::-1]
    for yi, (lab, v, h) in zip(y, ARC):
        ax.barh(yi, v, color="0.55" if not h else "0.85", edgecolor="0.1",
                hatch=h, height=0.7)
        ax.text(v - 0.004, yi, f"{v:.4f}", va="center", ha="right", fontsize=7,
                color="white" if not h else "black",
                bbox=dict(fc="white", ec="none", pad=0.6) if h else None)
    ax.set_yticks(y)
    ax.set_yticklabels([a[0] for a in ARC])
    ax.set_xlim(0.7, 1.0)
    ax.set_xlabel("First-5s validation macro AUC")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "diagnostic_arc")


def lb():
    fig, ax = plt.subplots(figsize=SIZE)
    w = 0.38
    xs = range(len(LB))
    ax.bar([x - w / 2 for x in xs], [r[1] for r in LB], w, color="0.35",
           edgecolor="0.1", label="Public")
    ax.bar([x + w / 2 for x in xs], [r[2] for r in LB], w, color="0.85",
           edgecolor="0.1", hatch="//", label="Private")
    ax.hlines(CUTOFF, -0.6, 3.45, color="0.1", ls="--", lw=1.0)
    ax.hlines(GOAL, -0.6, 3.45, color="0.1", ls="-.", lw=1.0)
    ax.text(3.5, CUTOFF, "0.776", ha="left", va="center", fontsize=7)
    ax.text(3.5, GOAL, "0.810", ha="left", va="center", fontsize=7)
    ax.set_xlim(-0.6, 4.0)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r[0] for r in LB])
    ax.set_ylim(0.74, 0.87)
    ax.set_ylabel("Leaderboard macro AUC")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "leaderboard_compare")


def gap():
    names = ["A1", "A2", "A3"]
    d = {r[0]: r for r in LB}
    pub = [VAL[n] - d[n][1] for n in names]
    pri = [VAL[n] - d[n][2] for n in names]
    fig, ax = plt.subplots(figsize=SIZE)
    w = 0.38
    xs = range(3)
    ax.bar([x - w / 2 for x in xs], pub, w, color="0.35", edgecolor="0.1",
           label="Public")
    ax.bar([x + w / 2 for x in xs], pri, w, color="0.85", edgecolor="0.1",
           hatch="//", label="Private")
    for x, a, b in zip(xs, pub, pri):
        ax.text(x - w / 2, a + 0.003, f"{a:.4f}", ha="center", fontsize=6.5)
        ax.text(x + w / 2, b + 0.003, f"{b:.4f}", ha="center", fontsize=6.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 0.25)
    ax.set_ylabel("Val AUC minus leaderboard AUC")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "transfer_gap")


if __name__ == "__main__":
    arc(); lb(); gap()
    for n in ["A1", "A2", "A3"]:
        d = {r[0]: r for r in LB}[n]
        print(n, f"{VAL[n]-d[1]:.4f} {VAL[n]-d[2]:.4f}")
