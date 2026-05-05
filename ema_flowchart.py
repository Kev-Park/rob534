#!/usr/bin/env python3
"""
Flowchart: how the EMA struggle score is calculated and interrupt decided.
Run: python rob534/ema_flowchart.py
Saves: rob534/ema_flowchart.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Color palette ────────────────────────────────────────────────────────────
C = {
    "frames":   dict(face="#7D3C98", edge="#4A235A", text="white"),   # purple
    "gemini":   dict(face="#B03A2E", edge="#7B241C", text="white"),   # red
    "panel":    dict(face="#CA6F1E", edge="#784212", text="white"),   # orange
    "median":   dict(face="#2E86C1", edge="#1A5276", text="white"),   # blue
    "ema":      dict(face="#1E8449", edge="#145A32", text="white"),   # green
    "decision": dict(face="#D4AC0D", edge="#9A7D0A", text="black"),   # yellow
    "output":   dict(face="#138D75", edge="#0E6655", text="white"),   # teal
    "note":     dict(face="#ECF0F1", edge="#BDC3C7", text="#2C3E50"), # light grey
    "start_end":dict(face="#1C2833", edge="#0D1117", text="white"),   # near-black
}

FIG_W, FIG_H = 18, 28
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("#F8F9FA")


# ── Drawing helpers ──────────────────────────────────────────────────────────
def rect(x, y, w, h, label, col, fontsize=9, sub=None):
    c = C[col]
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.12",
                       facecolor=c["face"], edgecolor=c["edge"],
                       linewidth=2, zorder=3)
    ax.add_patch(p)
    top_y = y + (0.15 if sub else 0)
    ax.text(x, top_y, label, ha="center", va="center",
            fontsize=fontsize, color=c["text"], fontweight="bold",
            zorder=4, multialignment="center")
    if sub:
        ax.text(x, y - h * 0.22, sub, ha="center", va="center",
                fontsize=fontsize - 1.5, color=c["text"], style="italic",
                zorder=4, multialignment="center")


def diamond(x, y, w, h, label, col, fontsize=9):
    c = C[col]
    pts = np.array([[x, y+h/2], [x+w/2, y], [x, y-h/2], [x-w/2, y]])
    patch = plt.Polygon(pts, closed=True,
                        facecolor=c["face"], edgecolor=c["edge"],
                        linewidth=2, zorder=3)
    ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color=c["text"], fontweight="bold",
            zorder=4, multialignment="center")


def oval(x, y, w, h, label, col, fontsize=10):
    c = C[col]
    p = mpatches.Ellipse((x, y), w, h,
                         facecolor=c["face"], edgecolor=c["edge"],
                         linewidth=2, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color=c["text"], fontweight="bold", zorder=4)


def arr(x1, y1, x2, y2, label="", color="#444", lw=2.0, style="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=style), zorder=2)
    if label:
        mx, my = (x1+x2)/2 + 0.15, (y1+y2)/2
        ax.text(mx, my, label, fontsize=8, color=color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


def note(x, y, w, h, text, fontsize=8):
    c = C["note"]
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.12",
                       facecolor=c["face"], edgecolor=c["edge"],
                       linewidth=1.2, linestyle="--", zorder=3)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=c["text"], zorder=4, multialignment="center")


def section_banner(y, label, color="#BDC3C7"):
    ax.axhline(y, color=color, lw=1, linestyle=":", zorder=1)
    ax.text(0.3, y + 0.1, label, fontsize=8, color="#7F8C8D",
            style="italic", va="bottom", zorder=5)


# ── Layout ───────────────────────────────────────────────────────────────────
CX  = 8.0    # main flow centre
BW  = 6.0    # box width
BH  = 0.8    # box height
NX  = 14.5   # note column x

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(CX, FIG_H - 0.5, "EMA Struggle Score Pipeline\n& Interrupt Decision",
        ha="center", fontsize=14, fontweight="bold", color="#1C2833")

# ═════════════════════════════════════════════════════════════════════════════
# INPUT: Camera frames at 30 Hz
# ═════════════════════════════════════════════════════════════════════════════
section_banner(FIG_H - 1.5, "INPUT  (continuous at 30 Hz)")

Y = FIG_H - 2.3
rect(CX, Y, BW, BH, "Camera frames  @30 Hz", "frames",
     sub="push_frame() fills a rolling deque (buffer_seconds * fps)")

note(NX, Y, 4.8, 0.7,
     "buffer = deque(maxlen=600)\n= 20 s at 30 fps")

arr(CX, Y - BH/2, CX, Y - 1.15)
Y -= 1.5

rect(CX, Y, BW, BH, "Subsample N frames from buffer", "frames",
     sub="evenly spaced (default N=12)")

note(NX, Y, 4.8, 0.6,
     "e.g. 600 frames -> 12 evenly\nspaced snapshots of 20 s")

# ═════════════════════════════════════════════════════════════════════════════
# PANEL OF JUDGES
# ═════════════════════════════════════════════════════════════════════════════
arr(CX, Y - BH/2, CX, Y - 1.15)
Y -= 1.5
section_banner(Y + 0.55, "GEMINI PANEL  (every check_interval seconds)")

rect(CX, Y, BW + 1.0, BH + 0.25, "assess_interrupt_panel()", "panel",
     sub="5 parallel Gemini calls, same STRUGGLE_PROMPT, different temperatures")

# Fan out to judges
JY = Y - 1.5
temps = [0.10, 0.25, 0.40, 0.55, 0.70]
jx_start = CX - 3.6
jx_step  = 1.8
for i, t in enumerate(temps):
    jx = jx_start + i * jx_step
    arr(CX, Y - (BH + 0.25)/2, jx, JY + 0.35, color="#CA6F1E")
    # small judge box
    c = C["gemini"]
    p = FancyBboxPatch((jx - 0.75, JY - 0.35), 1.5, 0.7,
                       boxstyle="round,pad=0.08",
                       facecolor=c["face"], edgecolor=c["edge"],
                       linewidth=1.5, zorder=3)
    ax.add_patch(p)
    ax.text(jx, JY + 0.07, f"T={t:.2f}", ha="center", va="center",
            fontsize=8, color=c["text"], fontweight="bold", zorder=4)
    ax.text(jx, JY - 0.15, "Gemini", ha="center", va="center",
            fontsize=7, color=c["text"], zorder=4)

note(NX, JY, 4.8, 0.9,
     "Each judge returns:\n"
     "  struggling: bool\n"
     "  confidence: 0-1\n"
     "  reason: str")

# Converge to vote aggregation
Y = JY - 1.1
for i in range(5):
    jx = jx_start + i * jx_step
    arr(jx, JY - 0.35, CX, Y + BH/2, color="#CA6F1E")

rect(CX, Y, BW, BH, "Vote aggregation", "panel",
     sub="p_raw = vote_count / n_judges")

note(NX, Y, 4.8, 0.8,
     "e.g. 3 of 5 vote 'struggling'\n"
     "p_raw = 3/5 = 0.60\n"
     "Failed judges are excluded")

# ═════════════════════════════════════════════════════════════════════════════
# MEDIAN FILTER
# ═════════════════════════════════════════════════════════════════════════════
arr(CX, Y - BH/2, CX, Y - 1.15)
Y -= 1.5
section_banner(Y + 0.55, "SMOOTHING  (kills noise, builds confidence)")

rect(CX, Y, BW, BH + 0.1, "Rolling median filter", "median",
     sub="window = median_window (default 3)")

note(NX, Y, 4.8, 1.1,
     "History: [0.10, 0.10, 0.60]\n"
     "Sorted:  [0.10, 0.10, 0.60]\n"
     "Median:   0.10\n"
     "One spike can't move the needle")

# ═════════════════════════════════════════════════════════════════════════════
# EMA
# ═════════════════════════════════════════════════════════════════════════════
arr(CX, Y - (BH + 0.1)/2, CX, Y - 1.25)
Y -= 1.6

rect(CX, Y, BW, BH + 0.2, "Exponential Moving Average (EMA)", "ema",
     sub="score = alpha * p_median  +  (1 - alpha) * score_prev")

note(NX, Y, 4.8, 1.1,
     "alpha = ema_alpha (default 0.4)\n"
     "Slow rise: needs sustained\n"
     "agreement across multiple\n"
     "consecutive checks")

# ═════════════════════════════════════════════════════════════════════════════
# DECISION
# ═════════════════════════════════════════════════════════════════════════════
arr(CX, Y - (BH + 0.2)/2, CX, Y - 1.3)
Y -= 1.7
section_banner(Y + 0.65, "INTERRUPT DECISION")

diamond(CX, Y, 6.5, 1.2, "EMA score  >=  threshold ?", "decision", fontsize=10)

note(NX, Y, 4.8, 0.8,
     "interrupt_threshold (default 0.20)\n"
     "is_struggling() returns True\n"
     "when EMA crosses this line")

# YES branch
arr(CX + 3.25, Y, CX + 5.0, Y, label="YES", color="#B03A2E", lw=2.5)
YES_Y = Y
rect(CX + 7.2, Y, 3.8, BH + 0.3, "INTERRUPT", "gemini",
     fontsize=11, sub="switch to teacher policy B")

# NO branch
arr(CX, Y - 1.2/2, CX, Y - 1.3, label="NO", color="#1E8449", lw=2.5)
Y -= 1.7

rect(CX, Y, BW - 1.0, BH, "Continue policy A", "ema", fontsize=10)

# Loop-back arrow
arr(CX - (BW-1.0)/2, Y, CX - 4.5, Y, color="#888")
ax.annotate("", xy=(CX - 4.5, FIG_H - 2.3 - 0.4), xytext=(CX - 4.5, Y),
            arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.5,
                            linestyle="--"), zorder=2)
ax.text(CX - 4.8, (Y + FIG_H - 2.3) / 2, "wait check_interval\n(default 20 s)",
        fontsize=8, color="#888", rotation=90, va="center", ha="center")

# ═════════════════════════════════════════════════════════════════════════════
# LEGEND
# ═════════════════════════════════════════════════════════════════════════════
legend_items = [
    ("Camera / Frames",   "frames"),
    ("Gemini API",        "gemini"),
    ("Panel / Aggregation","panel"),
    ("Median Filter",     "median"),
    ("EMA / Continue",    "ema"),
    ("Decision",          "decision"),
]
lx, ly = 0.5, 3.5
ax.text(lx, ly + 0.3, "LEGEND", fontsize=9, fontweight="bold", color="#2C3E50")
for lbl, col in legend_items:
    c = C[col]
    p = FancyBboxPatch((lx, ly - 0.26), 0.6, 0.38,
                       boxstyle="round,pad=0.05",
                       facecolor=c["face"], edgecolor=c["edge"],
                       linewidth=1.5, zorder=3)
    ax.add_patch(p)
    ax.text(lx + 0.85, ly - 0.06, lbl, fontsize=8, va="center", color="#2C3E50")
    ly -= 0.5


# ── Save ─────────────────────────────────────────────────────────────────────
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ema_flowchart.png")
plt.tight_layout(pad=0.5)
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")

try:
    import subprocess, sys
    if sys.platform == "win32":
        subprocess.Popen(["start", out], shell=True)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", out])
    else:
        subprocess.Popen(["xdg-open", out])
except Exception:
    pass
