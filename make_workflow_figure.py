"""
Generate a workflow diagram for the data augmentation + phase analysis pipeline.
Annotates each step with the video files and sizes produced.

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
C_EDGE   = "#2C3E50"
C_TAG    = "#F8F9FA"   # file-size tag background
C_TAG_E  = "#7F8C8D"   # file-size tag border
C_BG     = "#FDFEFE"

FIG_W, FIG_H = 18, 14


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


def arr(ax, x0, y0, x1, y1, lw=1.6):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C_EDGE,
                                lw=lw, mutation_scale=15,
                                connectionstyle="arc3,rad=0.0"),
                zorder=2)


def elbow(ax, x0, y0, x1, y1, lw=1.6):
    """Vertical then horizontal elbow arrow."""
    ax.plot([x0, x0, x1], [y0, y1, y1], color=C_EDGE, lw=lw, zorder=2)
    ax.annotate("", xy=(x1, y1), xytext=(x0, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_EDGE,
                                lw=lw, mutation_scale=15),
                zorder=2)


def file_tag(ax, x, y, lines, align="left"):
    """Small file-size annotation tag."""
    ha = "left" if align == "left" else "right"
    txt = "\n".join(lines)
    ax.text(x, y, txt,
            ha=ha, va="center",
            fontsize=7.2, color="#4A4A4A",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3",
                      fc=C_TAG, ec=C_TAG_E, lw=0.8, alpha=0.95),
            zorder=5)


# ── layout ─────────────────────────────────────────────────────────────────────
Y1  = 12.8   # source datasets
Y2  = 11.0   # combine step
Y3  =  9.2   # combined dataset
Y4  =  7.2   # branch step 1
Y5  =  5.2   # branch step 2
Y6  =  3.2   # branch step 3
Y7  =  1.4   # final HF output (aug)

X_L  =  4.0   # augmentation column
X_R  = 14.0   # phase-analysis column
X_C  =  9.0   # centre

BH   = 1.0
BW   = 3.8
SBH  = 1.15


# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor(C_BG)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1 — two source HF datasets
# ═══════════════════════════════════════════════════════════════════════════════
SX1, SX2 = 5.8, 12.2

box(ax, SX1, Y1, 4.8, BH,
    ["nc8304/so101_032326_white_back_041026_cube_10",
     "HuggingFace  ·  50 episodes  ·  38 050 frames"],
    C_HF, fontsize=9)

box(ax, SX2, Y1, 4.8, BH,
    ["nc8304/so101_032326_white_back_040126_cube_13",
     "HuggingFace  ·  30 episodes  ·  22 954 frames"],
    C_HF, fontsize=9)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2 — combine step
# ═══════════════════════════════════════════════════════════════════════════════
arr(ax, SX1, Y1 - BH/2,  X_C - 0.6, Y2 + BH/2)
arr(ax, SX2, Y1 - BH/2,  X_C + 0.6, Y2 + BH/2)

box(ax, X_C, Y2, 4.6, BH,
    ["lerobot-edit-dataset  merge",
     "hugging_face_data_push.py  combine"],
    C_PROC, fontsize=9)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3 — combined dataset  +  file-size tag
# ═══════════════════════════════════════════════════════════════════════════════
arr(ax, X_C, Y2 - BH/2,  X_C, Y3 + BH/2)

box(ax, X_C, Y3, 5.4, BH,
    ["nc8304/so101_combined_cubeONLY",
     "80 episodes  ·  AV1 video  +  motor parquet"],
    C_HF, fontsize=9.5)

# file tag — right of combined box
file_tag(ax, X_C + 5.4/2 + 0.2, Y3,
         ["file-000.mp4  179 MB  (38 553 frames)",
          "file-001.mp4  121 MB  (23 080 frames)",
          "─────────────────────────────",
          "Total  300 MB  ·  61 633 frames"],
         align="left")

# ─── branch labels + divider ──────────────────────────────────────────────────
ax.text(X_L, Y3 - BH/2 - 0.22, "Color-Augmentation Pipeline",
        ha="center", va="top", fontsize=10, fontweight="bold", color="#1A5276")
ax.text(X_R, Y3 - BH/2 - 0.22, "Phase-Analysis Pipeline",
        ha="center", va="top", fontsize=10, fontweight="bold", color="#6C3483")
ax.axvline(X_C, ymin=0.04, ymax=0.60,
           color="#BDC3C7", lw=1.0, linestyle="--", zorder=1)

# ═══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION BRANCH  (left)
# ═══════════════════════════════════════════════════════════════════════════════

# elbow: combined left edge → batch_recolor
elbow(ax, X_C - 5.4/2, Y3,  X_L + BW/2, Y4)

box(ax, X_L, Y4, BW, SBH,
    ["batch_recolor.py",
     "segment orange cube  →  per-frame mask",
     "13 colours  ×  3 alphas  =  39 variants"],
    C_PROC, fontsize=8.8)

# file tag — left of aug column
file_tag(ax, X_L - BW/2 - 0.15, Y4,
         ["file-000/ recolored/  ×39 AVIs  7.5 GB",
          "file-001/ recolored/  ×39 AVIs  5.1 GB",
          "────────────────────────────────",
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
          "────────────────────────",
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

elbow(ax, X_C + 5.4/2, Y3,  X_R - BW/2, Y4)

box(ax, X_R, Y4, BW, SBH,
    ["batch_detect_phases.py",
     "gripper signal  ·  threshold crossing",
     "longest-hold cycle  →  pickup / drop frame"],
    C_PROC, fontsize=8.8)

# reads the same combined-dataset videos
file_tag(ax, X_R + BW/2 + 0.15, Y4,
         ["reads:",
          "  file-000.mp4  179 MB",
          "  file-001.mp4  121 MB",
          "(motor parquet only — no video decode)"],
         align="left")

arr(ax, X_R, Y4 - SBH/2,  X_R, Y5 + BH/2)

box(ax, X_R, Y5, BW, BH,
    ["episode_phases.parquet",
     "pickup_frame  ·  drop_frame  ·  hold_frames"],
    C_FILE, fontsize=8.8)

file_tag(ax, X_R + BW/2 + 0.15, Y5,
         ["episode_phases.parquet  <1 MB",
          "80 rows  ·  7 columns"],
         align="left")

arr(ax, X_R, Y5 - BH/2,  X_R, Y6 + SBH/2)

# two viewer boxes side by side
VW = 1.75
box(ax, X_R - VW/2 - 0.12, Y6, VW, SBH,
    ["phase_viewer.py",
     "PICKUP | HOLD | DROP",
     "episode slider"],
    C_VIZ, fontsize=8.2)

box(ax, X_R + VW/2 + 0.12, Y6, VW, SBH,
    ["episode_labeler.py",
     "frame-by-frame",
     "gripper plot overlay"],
    C_VIZ, fontsize=8.2)

file_tag(ax, X_R + BW/2 + 0.15, Y6,
         ["decodes frames via PyAV:",
          "  file-000.mp4  179 MB  (AV1)",
          "  file-001.mp4  121 MB  (AV1)"],
         align="left")

# ═══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════════
legend_handles = [
    mpatches.Patch(facecolor=C_HF,   edgecolor=C_EDGE, label="HuggingFace dataset / repo"),
    mpatches.Patch(facecolor=C_PROC, edgecolor=C_EDGE, label="Processing script"),
    mpatches.Patch(facecolor=C_FILE, edgecolor=C_EDGE, label="Local data file"),
    mpatches.Patch(facecolor=C_VIZ,  edgecolor=C_EDGE, label="Visualisation / review tool"),
    mpatches.Patch(facecolor=C_TAG,  edgecolor=C_TAG_E, label="Video files & sizes at each step"),
]
ax.legend(handles=legend_handles, loc="lower right",
          fontsize=8.5, framealpha=0.95, edgecolor=C_EDGE,
          bbox_to_anchor=(0.998, 0.002))

fig.suptitle("Data Pipeline: Color Augmentation & Episode Phase Labelling",
             fontsize=13, fontweight="bold", y=0.995)

out_png = OUT_DIR / "workflow_figure.png"
out_pdf = OUT_DIR / "workflow_figure.pdf"
fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=C_BG)
fig.savefig(out_pdf, bbox_inches="tight", facecolor=C_BG)
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
os.startfile(str(out_png))
