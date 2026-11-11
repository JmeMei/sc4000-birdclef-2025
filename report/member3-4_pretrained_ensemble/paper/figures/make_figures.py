"""Build all paper figures (vector PDF + 300-dpi PNG) into this folder.

Reads small local files only:
  report/member3-4_pretrained_ensemble/_kaggle_logs/_writeup/submissions.csv   (leaderboard scores by ref)
  report/member3-4_pretrained_ensemble/_kaggle_logs/{a1-v2s-train,ns-student-main,ns-student-ablation}/history.csv
  report/member3-4_pretrained_ensemble/_kaggle_logs/ens-e4/timing_profile.json
Val AUCs and timings that are not in those files are hard-coded below with sources.
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "lines.linewidth": 0.8, "hatch.linewidth": 0.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
    "legend.frameon": False,
})

HERE = Path(__file__).resolve().parent
LOGS = HERE.parents[1] / "_kaggle_logs"
ACC, ACC_T = "#4C6A92", "#DCE3EC"
DARK, MID, LIGHT = "#333333", "#808080", "#D9D9D9"
COL1, COLW = 3.3, 7.0


def save(fig, stem):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(HERE / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.01, **kw)
    plt.close(fig)
    print("wrote", stem)


# ---------------------------------------------------------------- data
SUB = {}
with open(LOGS / "_writeup" / "submissions.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        SUB[r["ref"]] = (float(r["publicScore"]), float(r["privateScore"]))


def lb(ref):
    return SUB[str(ref)]


V2 = (0.763, 0.789)  # baseline v2, rounded, not in submissions.csv


def private_bar_style(color_face=LIGHT, edge=DARK):
    return dict(facecolor=color_face, edgecolor=edge, hatch="////", linewidth=0.6)


# ------------------------------------------------ 1 leaderboard progression
def fig_lb_progression():
    items = [("v2\n(baseline)", V2), ("A1", lb(58120958)), ("A2", lb(58122219)),
             ("A3", lb(58127656)), ("A3 +\nPerch-\ndistilled", lb(58161782)),
             ("A2 +\nA1-\ndistilled", lb(58164973)), ("NS\nMAIN", lb(58181262)),
             ("NS\nABLATION", lb(58181267)), ("E2", lb(58196486)),
             ("E4\n(final)", lb(58198889))]
    fig, ax = plt.subplots(figsize=(COLW, 2.7))
    w = 0.38
    for i, (n, (pub, pri)) in enumerate(items):
        final = i == len(items) - 1
        ax.bar(i - w / 2, pub, w, facecolor=ACC if final else "#505050",
               edgecolor=DARK, linewidth=0.6, zorder=3)
        ax.bar(i + w / 2, pri, w, zorder=3,
               **private_bar_style(ACC_T if final else LIGHT, ACC if final else DARK))
        for dx, v in ((-w / 2, pub), (w / 2, pri)):
            ax.text(i + dx, v + 0.004, f"{v:.3f}", rotation=90, ha="center",
                    va="bottom", fontsize=7)
    ax.axhline(V2[0], color=DARK, ls=":", lw=0.7, zorder=2)
    ax.text(len(items) - 0.35, V2[0], "v2 public\n0.763", ha="left",
            va="center", fontsize=7, color=DARK, clip_on=False)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([n for n, _ in items], fontsize=7)
    ax.set_ylim(0.70, 0.93)
    ax.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90])
    ax.spines["left"].set_bounds(0.70, 0.93)
    ax.set_ylabel("ROC-AUC")
    ax.set_xlim(-0.6, len(items) - 0.4)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor="#505050", edgecolor=DARK, linewidth=0.6, label="Public"),
                       Patch(label="Private", **private_bar_style())],
              loc="upper left", ncol=2, handlelength=1.6, columnspacing=1.2)
    save(fig, "fig_lb_progression")


# ------------------------------------------- shared horizontal-bar helper
def grouped_hbars(fig_h, groups, xlabel, stem, xlim, fmt="{:+.3f}",
                  zero_line=False, xticks=None, legend_loc="lower right"):
    """groups: [(group_label, [(row_label, (a, b)), ...])]; a=public bar, b=private bar."""
    fig, ax = plt.subplots(figsize=(COL1, fig_h))
    y = 0.0
    ticks, labels, seps = [], [], []
    bh = 0.34
    span = xlim[1] - xlim[0]
    for gi, (g, rows) in enumerate(groups):
        if gi > 0:
            seps.append(y - 0.55)
        ax.text(xlim[0] + 0.005 * span, y, g, fontsize=7, fontweight="bold",
                color=DARK, va="center", ha="left")
        y += 0.8
        for lab, (a, b) in rows:
            for off, v, st in ((-bh / 2, a, 0), (bh / 2, b, 1)):
                kw = dict(facecolor="#505050", edgecolor=DARK, linewidth=0.6) if st == 0 \
                    else private_bar_style()
                ax.barh(y + off, v, bh, zorder=3, **kw)
                pad = 0.012 * span
                ax.text(v + (pad if v >= 0 else -pad), y + off, fmt.format(v),
                        va="center", ha="left" if v >= 0 else "right", fontsize=7)
            ticks.append(y)
            labels.append(lab)
            y += 1.0
        y += 0.15
    for s in seps:
        ax.axhline(s, color=MID, lw=0.5)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(y - 0.1, -0.5)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    if zero_line:
        ax.axvline(0, color="black", lw=0.8, zorder=4)
    ax.legend(handles=[Patch(facecolor="#505050", edgecolor=DARK, linewidth=0.6, label="Public"),
                       Patch(label="Private", **private_bar_style())],
              loc=legend_loc, handlelength=1.6)
    save(fig, stem)


def fig_adaptation_delta():
    def d(ref, base):
        p, q = lb(ref)
        return (p - base[0], q - base[1])
    A1, A2, A3 = lb(58120958), lb(58122219), lb(58127656)
    stats = [("A1 + AdaBN\n(vs A1)", d(58158788, A1)),
             ("v2 + AdaBN\n(vs v2)", d(58162933, V2)),
             ("A2 std-CORAL\n(vs A2)", d(58160573, A2)),
             ("A2 mean/std\n(vs A2)", d(58160577, A2)),
             ("A3 std-CORAL\n(vs A3)", d(58160580, A3)),
             ("A3 mean/std\n(vs A3)", d(58160584, A3))]
    mixed = [("Distill + CORAL (A3)\n(vs A3)", d(58162938, A3))]
    content = [("A3 Perch-distilled\n(vs A3)", d(58161782, A3)),
               ("A2 A1-distilled\n(vs A2)", d(58164973, A2)),
               ("NS MAIN\n(vs A1)", d(58181262, A1)),
               ("NS ABLATION\n(vs A1)", d(58181267, A1))]
    for rows in (stats, mixed, content):
        for n, (a, b) in rows:
            print(f"delta {n.splitlines()[0]:24s} pub {a:+.4f} pri {b:+.4f}")
    grouped_hbars(5.6, [("Statistics-level", stats), ("Mixed (statistics + content)", mixed),
                        ("Content-level", content)],
                  "Δ AUC vs unadapted model", "fig_adaptation_delta",
                  (-0.23, 0.09), zero_line=True, xticks=[-0.20, -0.15, -0.10, -0.05, 0, 0.05],
                  legend_loc="upper right")


def fig_transfer_gap():
    # source: report/member3-4_pretrained_ensemble/results/results_table.md and report/member3-4_pretrained_ensemble/LOG.md (val AUC); v2 from v2-adabn run cfg
    val = {"v2": 0.9766, "A1": 0.9832, "A2": 0.9916, "A3": 0.9749, "A1+AdaBN": 0.9713,
           "A2 std-CORAL": 0.9901, "A3 std-CORAL": 0.9728, "A3 Perch-distilled": 0.97504,
           "A2 + A1-distilled": 0.99182, "NS MAIN": 0.98650, "NS ABLATION": 0.98541}
    ref = {"v2": V2, "A1": lb(58120958), "A2": lb(58122219), "A3": lb(58127656),
           "A1+AdaBN": lb(58158788), "A2 std-CORAL": lb(58160573),
           "A3 std-CORAL": lb(58160580), "A3 Perch-distilled": lb(58161782),
           "A2 + A1-distilled": lb(58164973), "NS MAIN": lb(58181262),
           "NS ABLATION": lb(58181267)}

    def row(k):
        return (k, (val[k] - ref[k][0], val[k] - ref[k][1]))
    groups = [("Base models", [row(k) for k in ("v2", "A1", "A2", "A3")]),
              ("Statistics-level", [row(k) for k in ("A1+AdaBN", "A2 std-CORAL", "A3 std-CORAL")]),
              ("Content-level", [row(k) for k in ("A3 Perch-distilled", "A2 + A1-distilled",
                                                 "NS MAIN", "NS ABLATION")])]
    for g, rows in groups:
        for n, (a, b) in rows:
            print(f"gap {n:20s} val-pub {a:.4f} val-pri {b:.4f}")
    grouped_hbars(4.3, groups, "Transfer gap (val AUC − leaderboard AUC)",
                  "fig_transfer_gap", (0, 0.42), fmt="{:.3f}", xticks=[0, 0.1, 0.2, 0.3],
                  legend_loc="lower right")


def fig_timing():
    # source: report/member3-4_pretrained_ensemble/results/timing_table.md, report/member3-4_pretrained_ensemble/results/results_table.md
    rows = [("v2 profile (baseline pool)", 11.1, True), ("v2 fast pipeline", 5.5, True),
            ("A1 fast pipeline", 8.0, True), ("A1 submitted notebook", 37, False),
            ("A2 Perch", 25.0, False), ("A3 Bird-MAE", 61.6, False),
            ("A2+A1-distilled", 26.5, False), ("A3 Perch-distilled", 67.5, False),
            ("NS MAIN student", 9.9, True), ("E2", 30.4, True), ("E4 (final)", 32.3, True)]
    lab = lambda v: "37" if v == 37 else f"{v:.1f}"
    fig, ax = plt.subplots(figsize=(COL1, 3.2))
    for i, (n, v, meas) in enumerate(rows):
        final = n.startswith("E4")
        if meas:
            ax.barh(i, v, 0.62, facecolor=ACC if final else "#505050", edgecolor=DARK,
                    linewidth=0.6, zorder=3)
        else:
            ax.barh(i, v, 0.62, zorder=3, **private_bar_style())
        if v < 15:
            ax.text(v + 1, i, lab(v), va="center", ha="left", fontsize=7)
        else:
            ax.text(v - 1, i, lab(v), va="center", ha="right", fontsize=7, zorder=5,
                    color="white" if meas else "black",
                    bbox=None if meas else dict(fc="white", ec="none", pad=0.6))
    ax.axvline(75, color=DARK, ls="--", lw=0.7, zorder=2)
    ax.axvline(90, color="black", lw=0.9, zorder=2)
    ax.text(74, 9.5, "budget 75", rotation=90, ha="right", va="center", fontsize=7)
    ax.text(89, 9.5, "limit 90", rotation=90, ha="right", va="center", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, 95)
    ax.set_xlabel("CPU minutes per 700 soundscapes")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor="#505050", edgecolor=DARK, linewidth=0.6, label="Measured"),
                       Patch(label="Projected", **private_bar_style())],
              loc="upper right", bbox_to_anchor=(0.76, 1.0), handlelength=1.6)
    save(fig, "fig_timing")


def fig_training_curves():
    runs = [("A1", "a1-v2s-train", "-", "o"), ("NS MAIN", "ns-student-main", "--", "s"),
            ("NS ABLATION", "ns-student-ablation", ":", "^")]
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    for name, d, ls, mk in runs:
        ep, va = [], []
        with open(LOGS / d / "history.csv", newline="") as f:
            for r in csv.DictReader(f):
                ep.append(int(r["epoch"]))
                va.append(float(r["val_auc"]))
        b = max(range(len(va)), key=lambda i: va[i])
        print(f"curve {name}: best epoch {ep[b]} val_auc {va[b]:.4f}; final {va[-1]:.4f}")
        ax.plot(ep, va, color=DARK, ls=ls, marker=mk, ms=3, mfc="white", mew=0.6,
                lw=0.8, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation ROC-AUC")
    ax.set_xlim(0.5, 15.5)
    ax.set_xticks([1, 5, 10, 15])
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", handlelength=2.6)
    save(fig, "fig_training_curves")


# ------------------------------------------------------------ schematics
def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, kind="neutral", tag=None, fs=7):
    """kinds: neutral (grey, solid), frozen (white, dashed), trained (accent tint, solid),
    ghost / ghosttr (greyed, dashed), plain (white, solid)."""
    st = dict(neutral=("#EDEDED", DARK, "-", DARK), frozen=("white", DARK, (0, (3, 2)), DARK),
              trained=(ACC_T, ACC, "-", DARK), ghost=("white", MID, (0, (3, 2)), MID),
              plain=("white", DARK, "-", DARK), ghosttr=("white", MID, (0, (3, 2)), MID))[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=st[0], ec=st[1], ls=st[2], lw=0.7, zorder=2))
    if tag is None:
        tag = {"frozen": "frozen", "trained": "trained", "ghost": "frozen",
               "ghosttr": "trained"}.get(kind)
    if tag:
        ax.text(x + w / 2, y + h / 2 + 0.07, text, ha="center", va="center", fontsize=fs,
                color=st[3], zorder=3, linespacing=1.15)
        ax.text(x + w / 2, y + 0.04, tag, ha="center", va="bottom", fontsize=7, style="italic",
                color=MID if kind in ("frozen", "ghost", "ghosttr") else ACC, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=st[3], zorder=3, linespacing=1.15)


def arrow(ax, p, q, ls="-", color=DARK):
    ax.annotate("", xy=q, xytext=p, zorder=1,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=color, ls=ls, shrinkA=0,
                                shrinkB=0, mutation_scale=7))


def ft_legend(ax, x, y):
    box(ax, x, y, 0.32, 0.18, "", "frozen", tag="")
    ax.text(x + 0.38, y + 0.09, "frozen", fontsize=7, va="center")
    box(ax, x + 0.95, y, 0.32, 0.18, "", "trained", tag="")
    ax.text(x + 1.33, y + 0.09, "trained", fontsize=7, va="center")


def fig_f1():
    fig, ax = canvas(COLW, 3.5)
    rows = {"cnn": 2.95, "perch": 2.0, "mae": 1.05}
    h = 0.65
    box(ax, 0.05, 2.0 - 0.45, 0.85, 0.9, "5-s window\n(32 kHz\naudio)", "neutral")
    yc = rows["cnn"]
    box(ax, 1.05, yc - h / 2, 0.95, h, "Log-mel\nspectrogram\n(128 mels)", "neutral")
    box(ax, 2.2, yc - h / 2, 2.75, h, "A1 / noisy-student\nEfficientNetV2-S CNN", "trained",
        tag="trained (fine-tuned)")
    yc = rows["perch"]
    box(ax, 1.05, yc - h / 2, 0.95, h, "Raw 32 kHz\nwaveform", "neutral")
    box(ax, 2.2, yc - h / 2, 1.05, h, "Perch 2.0\nencoder", "frozen")
    box(ax, 4.05, yc - h / 2, 0.9, h, "MLP head", "trained")
    ax.text(3.65, yc + 0.05, "1536-d\nembedding", ha="center", va="bottom", fontsize=7, linespacing=1.1)
    yc = rows["mae"]
    box(ax, 1.05, yc - h / 2, 0.95, h, "Kaldi\nfbank", "ghost", tag="")
    box(ax, 2.2, yc - h / 2, 1.05, h, "Bird-MAE-Base\nencoder", "ghost")
    box(ax, 4.05, yc - h / 2, 0.9, h, "MLP head", "ghosttr")
    ax.text(3.65, yc + 0.05, "768-d\nmean-\npooled", ha="center", va="bottom", fontsize=7,
            color=MID, linespacing=1.1)
    for k, yc in rows.items():
        g = k == "mae"
        c = MID if g else DARK
        ls = (0, (3, 2)) if g else "-"
        arrow(ax, (0.9, 2.0), (1.05, yc), ls=ls, color=c)
        arrow(ax, (2.0, yc), (2.2, yc), ls=ls, color=c)
        if k != "cnn":
            arrow(ax, (3.25, yc), (4.05, yc), ls=ls, color=c)
        box(ax, 5.1, yc - 0.28, 0.7, 0.56, "206-class\nprobs", "ghost" if g else "neutral", tag="")
        arrow(ax, (4.95, yc), (5.1, yc), ls=ls, color=c)
        arrow(ax, (5.8, yc), (5.95, yc if not g else 1.6), ls=ls, color=c)
    box(ax, 5.95, 1.55, 1.0, 1.8, "Temporal\nsmoothing\n0.25/0.5/0.25\nwithin file", "plain")
    box(ax, 5.95, 0.8, 1.0, 0.58, "Per-class rank avg\n0.4 CNN +\n0.6 Perch", "plain", fs=6.8)
    box(ax, 5.95, 0.05, 1.0, 0.58, "submission.csv\n12 windows ×\n206 classes", "plain", fs=6.8)
    arrow(ax, (6.45, 1.55), (6.45, 1.38))
    arrow(ax, (6.45, 0.8), (6.45, 0.63))
    ax.text(0.05, 0.42, "Final E4 = CNN branch (0.4) + Perch branch (0.6).\n"
            "Bird-MAE branch (greyed, dashed): studied, not in final ensemble.",
            fontsize=7, va="center", color=DARK, linespacing=1.3)
    ft_legend(ax, 4.0, 0.33)
    save(fig, "fig_f1_architecture")


def fig_f2():
    fig, ax = canvas(COLW, 3.0)
    box(ax, 0.05, 2.0, 2.4, 0.95, "train_audio\nlabelled focal clips: 28,564\nsplit 80/20 (seed 42, stratified):\n"
        "22,851 train / 5,713 val", "neutral")
    box(ax, 0.05, 1.05, 2.4, 0.8, "train_soundscapes (unlabelled)\n1,000 files = 12,000 windows\n"
        "all 9,726 files = 116,712 windows", "neutral")
    box(ax, 0.05, 0.25, 2.4, 0.65, "hidden test\n~700 soundscapes\npublic / private leaderboard", "neutral")
    gx, gw, gh = 3.7, 3.25, 0.44
    cy = [2.68 - 0.5 * i for i in range(5)]
    texts = ["Base models A1 / A2 / A3\ninput: train_audio",
             "Statistics alignment (AdaBN, CORAL)\ninput: train_audio + 1,000 soundscapes",
             "Distillation (teacher labels)\ninput: train_audio + 1,000 soundscapes",
             "Noisy student\ninput: train_audio + 9,726 soundscapes",
             "Evaluation: val AUC on 5,713 clips (recorded, not used\nfor selection); selection by public LB on hidden test"]
    for y, t in zip(cy, texts):
        box(ax, gx, y - gh / 2, gw, gh, t, "plain")
    xa, xs_ = 3.5, 3.25
    ax.plot([2.45, xa], [cy[0], cy[0]], color=DARK, lw=0.7)
    ax.plot([xa, xa], [cy[0], cy[4] + 0.08], color=DARK, lw=0.7)
    for i, y in enumerate(cy):
        arrow(ax, (xa, y + (0.08 if i else 0)), (gx, y + (0.08 if i else 0)))
    dash = (0, (3, 2))
    ax.plot([2.45, xs_], [1.45, 1.45], color=MID, lw=0.7, ls=dash)
    ax.plot([xs_, xs_], [cy[1] - 0.08, cy[3] - 0.08], color=MID, lw=0.7, ls=dash)
    for y in cy[1:4]:
        ax.plot([xs_, xa], [y - 0.08, y - 0.08], color=MID, lw=0.7, ls=dash)
        arrow(ax, (xa, y - 0.08), (gx, y - 0.08), color=MID, ls=dash)
    ax.plot([2.45, 3.0], [cy[4] - 0.08] * 2, color=DARK, lw=0.7)
    arrow(ax, (3.0, cy[4] - 0.08), (gx, cy[4] - 0.08))
    ax.text(0.05, 0.02, "solid: labelled clips and hidden test;  dashed grey: unlabelled soundscapes",
            fontsize=7, va="bottom", color=MID)
    save(fig, "fig_f2_data_flow")


def fig_f3():
    fig, ax = canvas(COLW, 3.3)
    pw, gap = 2.15, 0.2
    xs = [0.05 + i * (pw + gap) for i in range(3)]
    for x0, x1, t in ((xs[0], xs[1] + pw, "Statistics-level"), (xs[2], xs[2] + pw, "Content-level")):
        ax.add_patch(FancyBboxPatch((x0, 2.98), x1 - x0, 0.3, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc=LIGHT, ec=DARK, lw=0.7))
        ax.text((x0 + x1) / 2, 3.13, t, ha="center", va="center", fontsize=8, fontweight="bold")
    for x, t in zip(xs, ["AdaBN", "CORAL", "Distillation"]):
        ax.text(x + pw / 2, 2.82, t, ha="center", va="center", fontsize=8, fontweight="bold")
    bw = 1.95

    def col(x, items):
        y = 2.65
        prev = None
        for text, kind, hh in items:
            box(ax, x + (pw - bw) / 2, y - hh, bw, hh, text, kind)
            if prev is not None:
                arrow(ax, (x + pw / 2, prev), (x + pw / 2, y))
            prev = y - hh
            y -= hh + 0.18
    col(xs[0], [("Soundscape windows (1,000 files):\nforward pass only, BN in\ntrain mode, no gradient", "frozen", 0.8),
                ("BN running mean / var\nre-estimated (weights\nunchanged)", "plain", 0.6),
                ("Adapted CNN\n(same weights,\nnew BN statistics)", "neutral", 0.6)])
    col(xs[1], [("Clip embeddings (source),\nsoundscape embeddings\n(target)", "neutral", 0.6),
                ("Whiten clips with source\ncov.; re-colour to soundscape\nmean / covariance", "plain", 0.65),
                ("Head retrained on\ntransformed clips", "trained", 0.65)])
    ax.text(xs[1] + pw / 2, 0.14, "test features: untransformed", fontsize=7, ha="center", style="italic")
    col(xs[2], [("Teacher model\n(original A1 or A2)", "frozen", 0.65),
                ("Soft pseudo-labels on\nsoundscape windows\np^1.3, values < 0.01 → 0", "plain", 0.65),
                ("Student head trained on\nlabelled clips + pseudo-\nlabelled windows (weight 0.5)", "trained", 0.85)])
    ft_legend(ax, xs[0] + 0.4, 0.04)
    save(fig, "fig_f3_adaptation_methods")


def fig_f4():
    fig, ax = canvas(COLW, 3.0)
    box(ax, 0.05, 2.2, 1.2, 0.5, "A1 (original)", "frozen", tag="")
    box(ax, 0.05, 1.55, 1.2, 0.5, "A2 Perch head\n(original)", "frozen", tag="")
    box(ax, 0.05, 0.9, 1.2, 0.55, "A2 + A1-distilled\nhead", "frozen", tag="")
    box(ax, 1.5, 1.7, 1.0, 0.8, "MAIN teacher\nmean(A1, A2)", "plain")
    box(ax, 1.5, 0.9, 1.0, 0.55, "ABLATION\nteacher", "plain")
    arrow(ax, (1.25, 2.45), (1.5, 2.3))
    arrow(ax, (1.25, 1.8), (1.5, 1.9))
    arrow(ax, (1.25, 1.17), (1.5, 1.17))
    box(ax, 2.75, 0.9, 1.0, 1.6, "Sharpen:\np^1.3,\nvalues < 0.01\n→ 0", "plain")
    arrow(ax, (2.5, 2.05), (2.75, 1.95))
    arrow(ax, (2.5, 1.17), (2.75, 1.3))
    box(ax, 4.0, 0.9, 1.0, 1.6, "Pseudo-labels\nfor 116,712\nsoundscape\nwindows", "neutral")
    arrow(ax, (3.75, 1.7), (4.0, 1.7))
    box(ax, 5.25, 0.9, 1.7, 1.6,
        "Fresh bird-pretrained\nEfficientNetV2-S student\n(not initialised from A1)\n"
        "focal clips + ~30k\nsoundscape windows /\nepoch (loss weight 0.5)", "trained")
    arrow(ax, (5.0, 1.7), (5.25, 1.7))
    ax.text(0.05, 0.7, "Student recipe: mixup p = 0.5, Beta(0.4, 0.4); drop-path 0.15; soundscape-noise "
            "augmentation;\nsoft-target BCE; 15 epochs. One student per teacher lane (MAIN, ABLATION), same recipe.",
            fontsize=7, ha="left", va="top", linespacing=1.3)
    box(ax, 5.25, 0.05, 1.7, 0.4, "CPU inference", "plain")
    arrow(ax, (6.1, 0.9), (6.1, 0.45))
    ax.text(0.05, 2.9, "Teachers", fontsize=7.5, fontweight="bold", va="top")
    ax.text(5.25, 2.9, "Student", fontsize=7.5, fontweight="bold", va="top")
    ft_legend(ax, 0.05, 0.02)
    save(fig, "fig_f4_noisy_student")


def fig_f5():
    T = json.load(open(LOGS / "ens-e4" / "timing_profile.json"))
    s = T["stages_s"]

    def m(v):
        return f"{v:.0f} s" if v >= 10 else f"{v:.1f} s"
    fig, ax = canvas(COLW, 3.0)
    ax.text(0.05, 2.9, f"{T['n_files_used']} files, {T['num_workers']} workers × "
            f"{T['threads_per_worker']} thread. Stage figures: CPU-s summed over 4 workers.",
            fontsize=7, va="top")
    h = 0.65
    yt, yb, ym = 2.2, 1.3, 1.75
    box(ax, 0.05, ym - h / 2, 0.95, h, f"Shared audio\ndecode\n{m(s['decode'])}", "neutral")
    box(ax, 1.25, yt - h / 2, 0.95, h, f"CNN log-mel\n{m(s['cnn_mel'])}", "neutral")
    box(ax, 2.45, yt - h / 2, 1.2, h, f"MAIN student\nforward\n{m(s['model_main'])}", "trained", tag="")
    box(ax, 1.25, yb - h / 2, 0.95, h, f"Perch\npreprocess\n{m(s['perch_preprocess'])}", "neutral")
    box(ax, 2.45, yb - h / 2, 1.2, h, f"Perch 2.0 ONNX\n{m(s['model_a2d'])}\n({s['model_a2d']/60:.0f} CPU-min)",
        "frozen", tag="")
    box(ax, 3.9, yb - h / 2, 0.95, h, f"A2d head\n{m(s['a2d_head'])}", "trained", tag="")
    box(ax, 5.1, ym - h / 2, 1.0, h, f"Smoothing +\nrank avg 0.4/0.6\n{m(s['post_processing'])}",
        "plain", fs=6.5)
    box(ax, 6.3, ym - 0.3, 0.65, 0.6, "submission\n.csv", "plain", fs=6.5)
    arrow(ax, (1.0, ym), (1.25, yt))
    arrow(ax, (1.0, ym), (1.25, yb))
    arrow(ax, (2.2, yt), (2.45, yt))
    arrow(ax, (2.2, yb), (2.45, yb))
    arrow(ax, (3.65, yb), (3.9, yb))
    arrow(ax, (4.85, yb), (5.1, ym - 0.1))
    arrow(ax, (3.65, yt), (5.1, ym + 0.1))
    arrow(ax, (6.1, ym), (6.3, ym))
    ax.text(1.25, yt + h / 2 + 0.08, "Branch 1: CNN", fontsize=7, style="italic", va="bottom")
    ax.text(1.25, yb + h / 2 + 0.05, "Branch 2: Perch", fontsize=7, style="italic", va="bottom")
    tot_pipe, tot_nb = s["total_pipeline"] / 60, s["total_notebook"] / 60
    x0, x1, yy = 0.5, 6.7, 0.4
    sc = (x1 - x0) / 90
    ax.add_patch(Rectangle((x0, yy), tot_nb * sc, 0.22, fc=ACC, ec=DARK, lw=0.6, zorder=3))
    ax.add_patch(Rectangle((x0, yy), tot_pipe * sc, 0.22, fc=ACC_T, ec=DARK, lw=0.6, zorder=4))
    ax.plot([x0 + 75 * sc] * 2, [yy - 0.1, yy + 0.32], color=DARK, ls="--", lw=0.7)
    ax.plot([x0 + 90 * sc] * 2, [yy - 0.1, yy + 0.32], color="black", lw=0.9)
    ax.text(x0 + 75 * sc - 0.04, yy - 0.12, "budget 75 min", fontsize=7, ha="right", va="top")
    ax.text(x0 + 90 * sc, yy - 0.12, "limit 90 min", fontsize=7, ha="right", va="top")
    ax.text(x0, yy - 0.12, "0", fontsize=7, ha="center", va="top")
    ax.text(x0, yy + 0.4,
            f"pipeline wall {s['total_pipeline']:.0f} s = {tot_pipe:.1f} min (light);  "
            f"whole notebook {s['total_notebook']:.0f} s = {tot_nb:.1f} min (dark)",
            fontsize=7, va="bottom", ha="left")
    ax.text(0.05, 0.0, f"Not shown: setup incl. SHA check {s['setup_incl_sha']:.0f} s, model load "
            f"{T['model_load_s']['main']:.0f} s + {T['model_load_s']['a2d']:.0f} s.",
            fontsize=7, va="bottom", color=MID)
    save(fig, "fig_f5_inference_pipeline")


if __name__ == "__main__":
    fig_lb_progression()
    fig_adaptation_delta()
    fig_transfer_gap()
    fig_timing()
    fig_training_curves()
    fig_f1()
    fig_f2()
    fig_f3()
    fig_f4()
    fig_f5()
