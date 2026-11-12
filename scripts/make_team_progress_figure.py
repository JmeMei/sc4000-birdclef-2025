"""Make report/figures/team_progress.png. Usage: python3 scripts/make_team_progress_figure.py report/figures/team_progress.png"""
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
# Public / private macro ROC-AUC from README experiment log and results table
steps = [
 ("Baseline\nEffNet-B0",        0.71681, 0.69908, "J"),
 ("Baseline v2\n(crop+noise+sec)", 0.76299, 0.78864, "J"),
 ("SED\n5 s chunks",            0.77298, 0.78740, "S"),
 ("A1 bird CNN\nEffNetV2-S",    0.82822, 0.81642, "A"),
 ("A2 Perch 2.0\n+ head",       0.84774, 0.85226, "A"),
 ("A2d\n(A1-distilled)",        0.86651, 0.85737, "A"),
 ("Noisy student\nCNN",         0.85985, 0.85185, "A"),
 ("E4 final\nensemble",         0.88749, 0.88029, "A"),
]
col = {"J": "#2a78d6", "S": "#eb6834", "A": "#1baf7a"}
who = {"J": "Jamie: baseline + ablations", "S": "Sudhishna: SED model", "A": "Arshiya: pretrained models, adaptation, ensemble"}
ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(11, 4.8), dpi=200)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
x = range(len(steps)); base = 0.68
for i, (lab, pub, priv, p) in enumerate(steps):
    ax.bar(i, pub - base, bottom=base, width=0.56, color=col[p], edgecolor="#fcfcfb", linewidth=2, zorder=2)
    ax.text(i, pub + 0.004, f"{pub:.3f}", ha="center", va="bottom", fontsize=9.5, color=ink, fontweight="bold")
    ax.plot(i + 0.36, priv, marker="D", ms=7, color=ink2, mec="#fcfcfb", mew=1.5, zorder=3)
ax.set_xticks(list(x)); ax.set_xticklabels([s[0] for s in steps], fontsize=9, color=ink2)
ax.set_ylim(base, 0.91); ax.set_ylabel("Macro ROC-AUC", color=ink2)
ax.yaxis.grid(True, color=grid, lw=0.8, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#b5b4ad")
ax.tick_params(colors=ink2, length=0)
handles = [Patch(color=col[k], label=who[k]) for k in "JSA"] + [
    Line2D([], [], marker="D", ls="", color=ink2, ms=7, label="Private leaderboard (bars = public)")]
ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9, labelcolor=ink2)
ax.set_title("Public leaderboard AUC, 0.717 → 0.887, one change at a time", loc="left", color=ink, fontsize=12, pad=10)
fig.tight_layout(); fig.savefig(sys.argv[1], facecolor=fig.get_facecolor())
