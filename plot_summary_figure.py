"""
Generate a publication-quality summary figure from the model eval tables.

Layout (4 panels):
  A  Success rate          B  S score distribution (per-episode box plots)
  C  Per-channel stability metrics (grouped bars, mean ± std across episodes)
  D  Execution times       (grouped bars with error bars)

Output: outputs/plots/model_eval_summary_figure.png

Usage
-----
    python plot_summary_figure.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

EP_CSV  = Path(__file__).parent / "outputs" / "model_eval_episodes.csv"
SUM_CSV = Path(__file__).parent / "outputs" / "model_eval_summary.csv"
OUT     = Path(__file__).parent / "outputs" / "plots" / "model_eval_summary_figure.png"

# ── style ─────────────────────────────────────────────────────────────────────

MODEL_ORDER = [
    "phase-split combined",
    "phase-split new-prompts",
    "aug",
]
MODEL_SHORT = {
    "phase-split combined":    "Phase-split\ncombined",
    "phase-split new-prompts": "Phase-split\nnew prompts",
    "aug":                     "Aug",
}
COLORS = {
    "phase-split combined":    "#1565C0",   # deep blue
    "phase-split new-prompts": "#2E7D32",   # deep green
    "aug":                     "#B71C1C",   # deep red
}
ALPHA_BAR  = 0.82
ALPHA_BOX  = 0.55
EDGE_COLOR = "white"

CH_NAMES = ["E_RMS_mean", "J_RMS_mean", "neg_SPARC_mean", "rho_HF_mean", "sigma_bar_mean"]
CH_LABELS = ["E_RMS\n(track err)", "J_RMS\n(jerk)", "neg_SPARC\n(smooth)",
             "rho_HF\n(HF pwr)", "sigma_bar\n(stall)"]

TIME_METRICS = [
    ("pickup_time_s", "Pickup"),
    ("hold_time_s",   "Hold"),
    ("drop_time_s",   "Drop"),
]


def _bar_group(ax, positions, values, errors, models, width=0.22, **kw):
    n = len(models)
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width
    bars = []
    for m, off, val, err in zip(models, offsets, values, errors):
        b = ax.bar(positions + off, val, width,
                   color=COLORS[m], alpha=ALPHA_BAR,
                   edgecolor=EDGE_COLOR, linewidth=0.6,
                   yerr=err, capsize=3,
                   error_kw={"elinewidth": 1.0, "ecolor": "#333333", "capthick": 1.0},
                   **kw)
        bars.append(b)
    return bars


def main():
    ep  = pd.read_csv(EP_CSV)
    sm  = pd.read_csv(SUM_CSV).set_index("model")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.42,
        wspace=0.30,
        left=0.07, right=0.97,
        top=0.91,  bottom=0.08,
        height_ratios=[1, 1],
    )
    ax_A = fig.add_subplot(gs[0, 0])   # success rate
    ax_B = fig.add_subplot(gs[0, 1])   # S distribution
    ax_C = fig.add_subplot(gs[1, 0])   # per-channel metrics
    ax_D = fig.add_subplot(gs[1, 1])   # execution times

    models = MODEL_ORDER

    # ── A: success rate ───────────────────────────────────────────────────────
    rates  = [sm.loc[m, "success_rate"] for m in models]
    n_eps  = [sm.loc[m, "n_labeled"]    for m in models]
    n_ok   = [sm.loc[m, "n_success"]    for m in models]
    colors = [COLORS[m] for m in models]
    x      = np.arange(len(models))

    bars = ax_A.bar(x, rates, color=colors, alpha=ALPHA_BAR,
                    edgecolor=EDGE_COLOR, linewidth=0.8, width=0.55)
    for bar, rate, ok, n in zip(bars, rates, n_ok, n_eps):
        ax_A.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.012,
                  f"{ok}/{n}\n({rate:.0%})",
                  ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax_A.set_xticks(x)
    ax_A.set_xticklabels([MODEL_SHORT[m] for m in models], fontsize=9)
    ax_A.set_ylim(0, 0.65)
    ax_A.set_ylabel("Success rate", fontsize=10)
    ax_A.set_title("A   Success rate", fontsize=11, fontweight="bold", loc="left")
    ax_A.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax_A.spines[["top", "right"]].set_visible(False)
    ax_A.grid(axis="y", alpha=0.3, linewidth=0.6)

    # ── B: S score box plots per model ────────────────────────────────────────
    bp_data = [ep[ep["model"] == m]["S_mean"].dropna().values for m in models]
    bp = ax_B.boxplot(
        bp_data,
        patch_artist=True,
        widths=0.45,
        medianprops={"color": "white", "linewidth": 2.0},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 3.5, "alpha": 0.5},
    )
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(ALPHA_BOX + 0.1)
        patch.set_edgecolor(COLORS[m])
    for flier, m in zip(bp["fliers"], models):
        flier.set_markerfacecolor(COLORS[m])
        flier.set_markeredgecolor(COLORS[m])

    # overlay individual points (jittered)
    rng = np.random.default_rng(42)
    for i, (data, m) in enumerate(zip(bp_data, models), start=1):
        jitter = rng.uniform(-0.15, 0.15, len(data))
        ax_B.scatter(i + jitter, data,
                     color=COLORS[m], alpha=0.45, s=18, zorder=3,
                     edgecolors="none")

    ax_B.axhline(1.0, color="gray", linewidth=0.8, linestyle="--",
                 alpha=0.7, label="P95 threshold = 1.0")
    ax_B.set_xticks(range(1, len(models) + 1))
    ax_B.set_xticklabels([MODEL_SHORT[m] for m in models], fontsize=9)
    ax_B.set_ylabel("S score (per-episode mean)", fontsize=10)
    ax_B.set_title("B   Stability score S  (0 = perfect, 1 = P95 threshold)",
                   fontsize=11, fontweight="bold", loc="left")
    ax_B.spines[["top", "right"]].set_visible(False)
    ax_B.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax_B.set_ylim(0, None)
    ax_B.legend(fontsize=8, framealpha=0.6)

    # ── C: per-channel stability metrics ─────────────────────────────────────
    x_ch = np.arange(len(CH_NAMES))
    vals = [[ep[ep["model"] == m][c].dropna().mean() for c in CH_NAMES]
            for m in models]
    errs = [[ep[ep["model"] == m][c].dropna().std() for c in CH_NAMES]
            for m in models]

    n_m = len(models)
    width = 0.22
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width
    for m, off, val, err in zip(models, offsets, vals, errs):
        ax_C.bar(x_ch + off, val, width,
                 color=COLORS[m], alpha=ALPHA_BAR,
                 edgecolor=EDGE_COLOR, linewidth=0.6,
                 yerr=err, capsize=3,
                 error_kw={"elinewidth": 1.0, "ecolor": "#444", "capthick": 1.0},
                 label=MODEL_SHORT[m].replace("\n", " "))

    ax_C.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_C.set_xticks(x_ch)
    ax_C.set_xticklabels(CH_LABELS, fontsize=9)
    ax_C.set_ylabel("Normalised value  (0–1, 1 = P95)", fontsize=10)
    ax_C.set_title("C   Per-channel stability metrics  (mean ± std across episodes)",
                   fontsize=11, fontweight="bold", loc="left")
    ax_C.set_ylim(0, 1.18)
    ax_C.spines[["top", "right"]].set_visible(False)
    ax_C.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax_C.legend(fontsize=8.5, framealpha=0.7, ncol=3,
                loc="upper right")

    # ── D: execution times ────────────────────────────────────────────────────
    t_labels = [label for _, label in TIME_METRICS]
    t_cols   = [col   for col, _  in TIME_METRICS]
    x_t  = np.arange(len(t_cols))
    t_vals = [[ep[ep["model"] == m][c].dropna().mean() for c in t_cols]
              for m in models]
    t_errs = [[ep[ep["model"] == m][c].dropna().std()  for c in t_cols]
              for m in models]

    for m, off, val, err in zip(models, offsets, t_vals, t_errs):
        ax_D.bar(x_t + off, val, width,
                 color=COLORS[m], alpha=ALPHA_BAR,
                 edgecolor=EDGE_COLOR, linewidth=0.6,
                 yerr=err, capsize=3,
                 error_kw={"elinewidth": 1.0, "ecolor": "#444", "capthick": 1.0})

    ax_D.set_xticks(x_t)
    ax_D.set_xticklabels(t_labels, fontsize=10)
    ax_D.set_ylabel("Time (s)", fontsize=10)
    ax_D.set_title("D   Execution times  (mean ± std, gripper-detected)",
                   fontsize=11, fontweight="bold", loc="left")
    ax_D.spines[["top", "right"]].set_visible(False)
    ax_D.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax_D.set_ylim(0, None)

    # shared legend for D
    legend_patches = [
        mpatches.Patch(facecolor=COLORS[m], alpha=ALPHA_BAR,
                       label=MODEL_SHORT[m].replace("\n", " "))
        for m in models
    ]
    ax_D.legend(handles=legend_patches, fontsize=8.5, framealpha=0.7, ncol=1,
                loc="upper left")

    # ── title ─────────────────────────────────────────────────────────────────
    fig.suptitle("SmolVLA model evaluation  —  3 training variants  (141 episodes)",
                 fontsize=13, fontweight="bold", y=0.975)

    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
