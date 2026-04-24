"""
Generate a workflow diagram for the data augmentation + phase analysis pipeline.
Shows how color-augmented dataset + episode_phases CSV converge for training.

Output: outputs/workflow_figure.pdf  (and .png)
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────────
C_HF     = "#AED6F1"   # HuggingFace datasets
C_PROC   = "#A9DFBF"   # processing scripts
C_FILE   = "#FAD7A0"   # local data / parquet
C_VIZ    = "#D7BDE2"   # visualisation tools
C_TRAIN  = "#F1948A"   # training node
C_EDGE   = "#2C3E50"
C_TAG    = "#F8F9FA"
C_TAG_E  = "#7F8C8D"
C_BG     = "#FDFEFE"

FIG_W, FIG_H = 20, 17


# ── helpers ────────────────────────────────────────────────────────────────────

def box(ax, cx, cy, w, h, lines, fc,
        fontsize=9, bold_first=True, radius=0.2, lw=1.4):
    patch = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=C_EDGE, facecolor=fc, zorder=3,
    )
    ax.add_patch(patch)
    n = len(lines)
    dy = h / (n + 1)
    for i, txt in enumerate(lines):
        weight = "bold" if (bold_first and i == 0) else "normal"
        fs = fontsize if i == 0 else fontsize - 0.5
        ax.text(cx, cy + h/2 - dy*(i+1), txt,
                ha="center", va="center",
                fontsize=fs, fontweight=weight,
                color="#1A1A1A", zorder=4)


def arr(ax, x0, y0, x1, y1, lw=1.6, color=C_EDGE):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=15,
                                connectionstyle="arc3,rad=0.0"),
                zorder=2)


def elbow(ax, x0, y0, x1, y1, lw=1.6, color=C_EDGE):
    """Go vertically from (x0,y0) to y1, then horizontally to (x1,y1)."""
    ax.plot([x0, x0, x1], [y0, y1, y1], color=color, lw=lw, zorder=2)
    ax.annotate("", xy=(x1, y1), xytext=(x0, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=15),
                zorder=2)


def file_tag(ax, x, y, lines, align="left"):
    ha = "left" if align == "left" else "right"
    txt = "\n".join(lines)
    ax.text(x, y, txt, ha=ha, va="center",
            fontsize=7.0, color="#4A4A4A", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3",
                      fc=C_TAG, ec=C_TAG_E, lw=0.8, alpha=0.95),
            zorder=5)


# ── layout ─────────────────────────────────────────────────────────────────────
Y1  = 15.8   # source datasets
Y2  = 14.0   # combine step
Y3  = 12.2   # combined dataset
Y4  = 10.1   # branch step 1
Y5  =  8.0   # branch step 2
Y6  =  5.9   # aug step 3 (merge_and_push)
Y7  =  4.0   # aug final (HF repo)
Y8  =  1.8   # TRAINING (centred)

X_L  =  4.2   # augmentation column
X_R  = 15.2   # phase-analysis column
X_C  =  9.7   # centre

BH   = 1.0
BW   = 4.0
SBH  = 1.2


# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor(C_BG)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1 — two source HF datasets
# ═══════════════════════════════════════════════════════════════════════════════
SX1, SX2 = 6.0, 13.5

box(ax, SX1, Y1, 5.2, BH,
    ["nc8304/so101_032326_white_back_041026_cube_10",
     "HuggingFace  ·  50 episodes  ·  38 050 frames"],
    C_HF, fontsize=9)

box(ax, SX2, Y1, 5.2, BH,
    ["nc8304/so101_032326_white_back_040126_cube_13",
     "HuggingFace  ·  30 episodes  ·  22 954 frames"],
    C_HF, fontsize=9)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2 — combine step
# ═══════════════════════════════════════════════════════════════════════════════
arr(ax, SX1, Y1 - BH/2,  X_C - 0.6, Y2 + BH/2)
arr(ax, SX2, Y1 - BH/2,  X_C + 0.6, Y2 + BH/2)

box(ax, X_C, Y2, 5.0, BH,
    ["lerobot-edit-dataset  merge",
     "hugging_face_data_push.py  combine"],
    C_PROC, fontsize=9)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3 — combined dataset
# ═══════════════════════════════════════════════════════════════════════════════
arr(ax, X_C, Y2 - BH/2,  X_C, Y3 + BH/2)

box(ax, X_C, Y3, 5.6, BH,
    ["nc8304/so101_combined_cubeONLY",
     "80 episodes  ·  AV1 video  +  motor parquet"],
    C_HF, fontsize=9.5)

file_tag(ax, X_C + 5.6/2 + 0.2, Y3,
         ["file-000.mp4  179 MB  (38 553 frames)",
          "file-001.mp4  121 MB  (23 080 frames)",
          "──────────────────────────────",
          "Total  300 MB  ·  61 633 frames"],
         align="left")

# branch labels + divider
ax.text(X_L, Y3 - BH/2 - 0.25, "Color-Augmentation Pipeline",
        ha="center", va="top", fontsize=10, fontweight="bold", color="#1A5276")
ax.text(X_R, Y3 - BH/2 - 0.25, "Phase-Analysis Pipeline",
        ha="center", va="top", fontsize=10, fontweight="bold", color="#6C3483")
ax.axvline(X_C, ymin=0.07, ymax=0.68,
           color="#BDC3C7", lw=1.0, linestyle="--", zorder=1)

# ═══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION BRANCH  (left)
# ═══════════════════════════════════════════════════════════════════════════════
elbow(ax, X_C - 5.6/2, Y3,  X_L + BW/2, Y4)

box(ax, X_L, Y4, BW, SBH,
    ["batch_recolor.py",
     "segment orange cube  →  per-frame mask",
     "13 colours  ×  3 alphas  =  39 variants"],
    C_PROC, fontsize=8.8)

file_tag(ax, X_L - BW/2 - 0.15, Y4,
         ["file-000/ recolored/  ×39 AVIs  7.5 GB",
          "file-001/ recolored/  ×39 AVIs  5.1 GB",
          "─────────────────────────────────",
          "Total  12.6 GB  (78 AVIs)"],
         align="right")

arr(ax, X_L, Y4 - SBH/2,  X_L, Y5 + SBH/2)

box(ax, X_L, Y5, BW, SBH,
    ["make_augmented_dataset.py",
     "AVI → AV1  ·  copy motor parquet",
     "39 variant datasets  ×  80 episodes"],
    C_PROC, fontsize=8.8)

file_tag(ax, X_L - BW/2 - 0.15, Y5,
         ["per variant:",
          "  file-000.mp4  ~550 MB",
          "  file-001.mp4  ~353 MB",
          "─────────────────────────",
          "39 variants  ×  ~900 MB  ≈  35 GB"],
         align="right")

arr(ax, X_L, Y5 - SBH/2,  X_L, Y6 + BH/2)

box(ax, X_L, Y6, BW, BH,
    ["merge_and_push.py",
     "aggregate  →  80 + 3 120 = 3 200 episodes"],
    C_PROC, fontsize=8.8)

file_tag(ax, X_L - BW/2 - 0.15, Y6,
         ["merged_dataset/  35 GB",
          "(orig + 39 variants)"],
         align="right")

arr(ax, X_L, Y6 - BH/2,  X_L, Y7 + BH/2)

box(ax, X_L, Y7, BW, BH,
    ["nc8304/so101_color_augmented",
     "HuggingFace Hub  ·  3 200 episodes"],
    C_HF, fontsize=9)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE-ANALYSIS BRANCH  (right)
# ═══════════════════════════════════════════════════════════════════════════════
elbow(ax, X_C + 5.6/2, Y3,  X_R - BW/2, Y4)

box(ax, X_R, Y4, BW, SBH,
    ["batch_detect_phases.py",
     "gripper signal  ·  threshold crossing",
     "longest-hold cycle  →  pickup / drop frame"],
    C_PROC, fontsize=8.8)

file_tag(ax, X_R + BW/2 + 0.15, Y4,
         ["reads motor parquet only:",
          "  file-000.mp4  179 MB",
          "  file-001.mp4  121 MB"],
         align="left")

arr(ax, X_R, Y4 - SBH/2,  X_R, Y5 + BH/2)

box(ax, X_R, Y5, BW, BH,
    ["episode_phases.csv  /  .parquet",
     "pickup_frame  ·  hold_frame  ·  drop_frame"],
    C_FILE, fontsize=8.8)

file_tag(ax, X_R + BW/2 + 0.15, Y5,
         ["80 rows  ·  same for all colour variants",
          "episode_phases.parquet  <1 MB"],
         align="left")

# ── viewers: side branch off episode_phases ───────────────────────────────────
VW = 1.85
VY = Y5 - 2.0   # place viewers below the parquet box
# connector line from parquet down to viewers row
ax.plot([X_R, X_R], [Y5 - BH/2, VY + SBH/2], color=C_EDGE, lw=1.2,
        linestyle=":", zorder=2)
ax.text(X_R, VY + SBH/2 + 0.08, "review / QC", ha="center", va="bottom",
        fontsize=7, color="#888888", style="italic")

box(ax, X_R - VW/2 - 0.12, VY, VW, SBH,
    ["phase_viewer.py",
     "PICKUP | HOLD | DROP",
     "episode slider"],
    C_VIZ, fontsize=8.2)

box(ax, X_R + VW/2 + 0.12, VY, VW, SBH,
    ["episode_labeler.py",
     "frame-by-frame",
     "gripper plot overlay"],
    C_VIZ, fontsize=8.2)

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING NODE  (centred at bottom)
# ═══════════════════════════════════════════════════════════════════════════════
TW, TH = 6.0, 1.3

box(ax, X_C, Y8, TW, TH,
    ["Training Dataset",
     "3 200 colour-augmented episodes  +  phase labels",
     "pickup_frame  ·  hold_frame  ·  drop_frame  per episode"],
    C_TRAIN, fontsize=9.2, radius=0.25, lw=2.0)

# arrow: so101_color_augmented → training
elbow(ax, X_L, Y7 - BH/2,  X_C - TW/2, Y8)

# arrow: episode_phases → training
elbow(ax, X_R, Y5 - BH/2,  X_C + TW/2, Y8)

# label on the converging arrows
ax.text(X_C - TW/2 - 0.15, (Y7 - BH/2 + Y8) / 2,
        "video\nepisodes", ha="right", va="center",
        fontsize=7.5, color="#1A5276", style="italic")
ax.text(X_C + TW/2 + 0.15, (Y5 - BH/2 + Y8) / 2,
        "phase\nlabels", ha="left", va="center",
        fontsize=7.5, color="#6C3483", style="italic")

# ═══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════════
legend_handles = [
    mpatches.Patch(facecolor=C_HF,    edgecolor=C_EDGE, label="HuggingFace dataset / repo"),
    mpatches.Patch(facecolor=C_PROC,  edgecolor=C_EDGE, label="Processing script"),
    mpatches.Patch(facecolor=C_FILE,  edgecolor=C_EDGE, label="Local data file"),
    mpatches.Patch(facecolor=C_VIZ,   edgecolor=C_EDGE, label="Review / visualisation tool"),
    mpatches.Patch(facecolor=C_TRAIN, edgecolor=C_EDGE, label="Training dataset (combined)"),
    mpatches.Patch(facecolor=C_TAG,   edgecolor=C_TAG_E, label="Video files & sizes"),
]
ax.legend(handles=legend_handles, loc="lower right",
          fontsize=8.5, framealpha=0.95, edgecolor=C_EDGE,
          bbox_to_anchor=(0.998, 0.002))

fig.suptitle("Data Pipeline: Color Augmentation & Phase-Labelled Training Dataset",
             fontsize=13, fontweight="bold", y=0.995)

out_png = OUT_DIR / "workflow_figure.png"
out_pdf = OUT_DIR / "workflow_figure.pdf"
fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=C_BG)
print(f"Saved: {out_png}")
try:
    fig.savefig(out_pdf, bbox_inches="tight", facecolor=C_BG)
    print(f"Saved: {out_pdf}")
except PermissionError:
    print(f"Skipped PDF (file open in viewer): {out_pdf}")
os.startfile(str(out_png))
