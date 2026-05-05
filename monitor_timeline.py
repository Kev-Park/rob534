#!/usr/bin/env python3
"""
Visual explanation of how LiveStruggleMonitor analyses a 90-second episode.
Run: python rob534/monitor_timeline.py
Saves: rob534/monitor_timeline.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Parameters (matching current defaults) ───────────────────────────────────
EPISODE_S     = 90       # episode duration
FPS           = 30       # camera fps
BUF_S         = 20       # rolling buffer window (seconds)
N_FRAMES      = 12       # frames subsampled per Gemini call
API_LATENCY   = 5.0      # seconds per Gemini call
CHECK_INTERVAL= 1.0      # check_interval arg (overridden by latency in practice)
EMA_ALPHA     = 0.4

# Effective call times (start of each Gemini response)
# First call possible when buf has N_FRAMES: t = N_FRAMES/FPS ≈ 0.4s
# But API takes 5s, so first result at ~5.4s, next at ~10.4s, etc.
first_result = N_FRAMES / FPS + API_LATENCY
call_times   = np.arange(first_result, EPISODE_S, API_LATENCY)

# Simulate a plausible p trajectory (starts low, rises mid-episode, drops at end)
np.random.seed(42)
def true_p(t):
    if t < 20:   return 0.10
    if t < 35:   return 0.15 + 0.02*(t-20)
    if t < 55:   return 0.45 + 0.15*np.sin((t-35)*np.pi/20)
    if t < 70:   return 0.70
    return 0.20

p_vals  = np.array([true_p(t) + np.random.normal(0, 0.04) for t in call_times])
p_vals  = np.clip(p_vals, 0.0, 1.0)

# Compute EMA
ema_vals = []
score = 0.0
for p in p_vals:
    score = EMA_ALPHA * p + (1 - EMA_ALPHA) * score
    ema_vals.append(score)
ema_vals = np.array(ema_vals)
THRESHOLD = 0.40

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor("#F8F9FA")

gs = fig.add_gridspec(4, 1, height_ratios=[1.4, 1, 1.8, 1.4],
                      hspace=0.55, top=0.93, bottom=0.06, left=0.08, right=0.97)

ax_buf  = fig.add_subplot(gs[0])   # rolling buffer timeline
ax_call = fig.add_subplot(gs[1])   # Gemini call events
ax_ema  = fig.add_subplot(gs[2])   # p + EMA over time
ax_info = fig.add_subplot(gs[3])   # payload diagram (frames sent to Gemini)

fig.suptitle("LiveStruggleMonitor — 90-second episode analysis",
             fontsize=14, fontweight="bold", color="#1C2833", y=0.97)

# ── Shared x-axis style ───────────────────────────────────────────────────────
for ax in [ax_buf, ax_call, ax_ema]:
    ax.set_xlim(0, EPISODE_S)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#FDFEFE")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1 — Rolling buffer
# ─────────────────────────────────────────────────────────────────────────────
ax_buf.set_title("Rolling 20-second frame buffer (600 frames max @ 30 fps)",
                 fontsize=10, color="#2C3E50", loc="left")
ax_buf.set_ylim(0, 1)
ax_buf.set_yticks([])
ax_buf.set_xlabel("Episode time (s)", fontsize=9)

# Shade the filled portion of the buffer at each moment
T = np.linspace(0, EPISODE_S, 500)
buf_fill = np.minimum(T, BUF_S)   # ramps up to BUF_S then stays flat

# Background: full possible buffer zone
ax_buf.fill_between(T, 0, 1, color="#EBF5FB", zorder=0)

# Filled buffer (relative fill fraction)
buf_frac = buf_fill / BUF_S
ax_buf.fill_between(T, 0, buf_frac, color="#2E86C1", alpha=0.35, zorder=1, label="frames in buffer")
ax_buf.plot(T, buf_frac, color="#2E86C1", lw=1.5, zorder=2)

# Annotate fill-up time
fill_t = BUF_S
ax_buf.axvline(fill_t, color="#2E86C1", lw=1.2, linestyle="--", alpha=0.6)
ax_buf.text(fill_t + 0.5, 0.85, f"buffer full\n({BUF_S}s)", fontsize=8,
            color="#2E86C1", va="top")

# Mark when first call is possible (N_FRAMES collected)
first_possible = N_FRAMES / FPS
ax_buf.axvline(first_possible, color="#E67E22", lw=1.2, linestyle=":")
ax_buf.text(first_possible + 0.4, 0.55,
            f"≥{N_FRAMES} frames\n→ first call\npossible ({first_possible:.1f}s)",
            fontsize=7.5, color="#E67E22", va="center")

# Frame count labels on right y-axis
ax_buf2 = ax_buf.twinx()
ax_buf2.set_ylim(0, 1)
ax_buf2.set_yticks([0, 0.5, 1.0])
ax_buf2.set_yticklabels(["0", "300", "600"], fontsize=8)
ax_buf2.set_ylabel("frames in buffer", fontsize=8, color="#2E86C1")
ax_buf2.spines["top"].set_visible(False)

ax_buf.set_ylabel("buffer fill", fontsize=9)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 2 — Gemini call events
# ─────────────────────────────────────────────────────────────────────────────
ax_call.set_title("Gemini API calls  (each call = 12 frames subsampled from buffer → ~5s latency)",
                  fontsize=10, color="#2C3E50", loc="left")
ax_call.set_ylim(0, 1)
ax_call.set_yticks([])
ax_call.set_xlabel("Episode time (s)", fontsize=9)

# Draw each call as a horizontal bar (start → result)
for i, t_result in enumerate(call_times):
    t_start  = t_result - API_LATENCY
    color    = "#E74C3C" if ema_vals[i] >= THRESHOLD else "#27AE60"
    # API call in flight
    ax_call.barh(0.5, API_LATENCY, left=t_start, height=0.28,
                 color="#BDC3C7", edgecolor="#95A5A6", linewidth=0.8, zorder=2)
    # Result marker
    ax_call.plot(t_result, 0.5, "D", color=color, ms=7, zorder=3)
    # p value label (every other call to avoid crowding)
    if i % 2 == 0:
        ax_call.text(t_result, 0.78, f"p={p_vals[i]:.2f}",
                     fontsize=7, ha="center", color=color, zorder=4)

# Legend
ax_call.plot([], [], "D", color="#E74C3C", ms=7, label="result: EMA ≥ threshold")
ax_call.plot([], [], "D", color="#27AE60", ms=7, label="result: ok")
ax_call.barh([], [], height=0.28, color="#BDC3C7", edgecolor="#95A5A6", label="Gemini call in flight (~5s)")
ax_call.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax_call.text(1, 0.1, "call in\nflight", fontsize=7.5, color="#95A5A6", va="bottom")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 3 — p + EMA chart
# ─────────────────────────────────────────────────────────────────────────────
ax_ema.set_title("interrupt_probability  p  (raw Gemini)  and  EMA score  (smoothed)",
                 fontsize=10, color="#2C3E50", loc="left")
ax_ema.set_ylim(-0.02, 1.05)
ax_ema.set_xlim(0, EPISODE_S)
ax_ema.set_xlabel("Episode time (s)", fontsize=9)
ax_ema.set_ylabel("score  (0–1)", fontsize=9)

# Threshold band
ax_ema.axhline(THRESHOLD, color="#E74C3C", lw=1.8, linestyle="--", zorder=2,
               label=f"interrupt_threshold = {THRESHOLD}")
ax_ema.fill_between([0, EPISODE_S], THRESHOLD, 1.05,
                    color="#FADBD8", alpha=0.4, zorder=0, label="STRUGGLING zone")

# Raw p (step-style — value holds until next call)
t_p = np.concatenate([[0], np.repeat(call_times, 2), [EPISODE_S]])
v_p = np.concatenate([[0], np.repeat(p_vals, 2)])
ax_ema.plot(t_p[:-1], v_p, color="#3498DB", lw=1.2, alpha=0.7,
            drawstyle="steps-post", label="p  (raw, per Gemini call)")
ax_ema.scatter(call_times, p_vals, color="#3498DB", s=30, zorder=4, alpha=0.8)

# EMA (step-style — holds between calls)
t_ema = np.concatenate([[0], np.repeat(call_times, 2), [EPISODE_S]])
v_ema = np.concatenate([[0], np.repeat(ema_vals, 2)])
ax_ema.plot(t_ema[:-1], v_ema, color="#E67E22", lw=2.5,
            drawstyle="steps-post", label=f"EMA  (α={EMA_ALPHA})", zorder=3)

# Mark where EMA crosses threshold
cross_idx = np.where(ema_vals >= THRESHOLD)[0]
if len(cross_idx):
    cross_t = call_times[cross_idx[0]]
    ax_ema.axvline(cross_t, color="#E74C3C", lw=2, linestyle="-", alpha=0.8, zorder=5)
    ax_ema.text(cross_t + 0.8, THRESHOLD + 0.04,
                f"AUTO-SWITCH\nt={cross_t:.0f}s", fontsize=9,
                color="#E74C3C", fontweight="bold", va="bottom")
    ax_ema.annotate("", xy=(cross_t, THRESHOLD),
                    xytext=(cross_t, THRESHOLD + 0.03),
                    arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.5))

# Freeze annotation between calls
mid = call_times[5]
ax_ema.annotate("EMA frozen\nbetween calls",
                xy=(mid + 1.5, ema_vals[5]),
                xytext=(mid + 5, ema_vals[5] - 0.12),
                fontsize=8, color="#E67E22",
                arrowprops=dict(arrowstyle="->", color="#E67E22", lw=1.2))

ax_ema.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
ax_ema.grid(axis="y", alpha=0.3)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 4 — Payload diagram: what Gemini actually receives
# ─────────────────────────────────────────────────────────────────────────────
ax_info.axis("off")
ax_info.set_xlim(0, 18)
ax_info.set_ylim(0, 3)
ax_info.set_title("What each Gemini call receives  (example at t=45s, buffer full)",
                  fontsize=10, color="#2C3E50", loc="left", pad=6)

# Buffer bar
buf_x, buf_y, buf_w, buf_h = 0.3, 1.6, 8.5, 0.6
ax_info.add_patch(FancyBboxPatch((buf_x, buf_y), buf_w, buf_h,
                                 boxstyle="round,pad=0.05",
                                 fc="#D6EAF8", ec="#2E86C1", lw=1.5))
ax_info.text(buf_x + buf_w/2, buf_y + buf_h + 0.15,
             "Rolling buffer  —  600 frames  (t=25s … t=45s)",
             ha="center", fontsize=9, color="#2E86C1", fontweight="bold")
ax_info.text(buf_x + 0.1, buf_y + buf_h/2, "oldest", fontsize=7.5,
             color="#7F8C8D", va="center")
ax_info.text(buf_x + buf_w - 0.1, buf_y + buf_h/2, "newest",
             fontsize=7.5, color="#7F8C8D", va="center", ha="right")

# 12 evenly-spaced sample markers on the buffer bar
sample_xs = np.linspace(buf_x + 0.3, buf_x + buf_w - 0.3, N_FRAMES)
for i, sx in enumerate(sample_xs):
    ax_info.plot(sx, buf_y + buf_h/2, "s",
                 color="#E67E22", ms=9, zorder=4)
    ax_info.plot([sx, sx], [buf_y, buf_y + buf_h], color="#E67E22",
                 lw=0.8, alpha=0.5, zorder=3)

ax_info.text(buf_x + buf_w/2, buf_y - 0.22,
             f"↑  {N_FRAMES} frames evenly subsampled  ↑",
             ha="center", fontsize=8.5, color="#E67E22")

# Arrow to Gemini box
arrow_x = buf_x + buf_w + 0.3
ax_info.annotate("", xy=(arrow_x + 0.7, buf_y + buf_h/2),
                 xytext=(arrow_x, buf_y + buf_h/2),
                 arrowprops=dict(arrowstyle="-|>", color="#566573", lw=2))
ax_info.text(arrow_x + 0.35, buf_y + buf_h/2 + 0.22,
             "send", fontsize=8, color="#566573", ha="center")

# Gemini box
gem_x = arrow_x + 0.8
ax_info.add_patch(FancyBboxPatch((gem_x, buf_y - 0.1), 2.8, buf_h + 0.2,
                                 boxstyle="round,pad=0.08",
                                 fc="#FDEDEC", ec="#E74C3C", lw=2))
ax_info.text(gem_x + 1.4, buf_y + buf_h/2 + 0.12,
             "Gemini", fontsize=10, ha="center",
             color="#E74C3C", fontweight="bold")
ax_info.text(gem_x + 1.4, buf_y + buf_h/2 - 0.18,
             "gemini-2.5-flash", fontsize=8, ha="center", color="#E74C3C")

# Arrow to output
out_x = gem_x + 2.8 + 0.3
ax_info.annotate("", xy=(out_x + 0.7, buf_y + buf_h/2),
                 xytext=(out_x, buf_y + buf_h/2),
                 arrowprops=dict(arrowstyle="-|>", color="#566573", lw=2))
ax_info.text(out_x + 0.35, buf_y + buf_h/2 + 0.22,
             "~5s", fontsize=8, color="#566573", ha="center")

# Output box
res_x = out_x + 0.8
ax_info.add_patch(FancyBboxPatch((res_x, buf_y - 0.28), 3.5, buf_h + 0.56,
                                 boxstyle="round,pad=0.08",
                                 fc="#EAFAF1", ec="#27AE60", lw=2))
ax_info.text(res_x + 1.75, buf_y + buf_h - 0.05,
             "Result", fontsize=9, ha="center", color="#1E8449", fontweight="bold")
for i, (k, v) in enumerate([("interrupt_probability", "0.45"),
                              ("pickup_attempts", "2"),
                              ("drop_attempts", "1"),
                              ("reason", '"struggling…"')]):
    ax_info.text(res_x + 0.15, buf_y + buf_h - 0.28 - i*0.22,
                 f"{k}: {v}", fontsize=7.5, color="#1E8449", va="top")

# EMA update annotation below
ax_info.text(res_x + 1.75, buf_y - 0.55,
             "EMA = 0.4 × 0.45 + 0.6 × prev_EMA", fontsize=8.5,
             ha="center", color="#E67E22",
             bbox=dict(boxstyle="round,pad=0.3", fc="#FEF9E7", ec="#F39C12", lw=1.5))

# Key stats bottom-right
stats_x = 14.5
ax_info.text(stats_x, 2.75, "Key numbers (defaults)", fontsize=9,
             fontweight="bold", color="#2C3E50")
rows = [
    ("Buffer window",      "20s  =  600 frames"),
    ("Frames to Gemini",   "12  (evenly spaced)"),
    ("API latency",        "~5s  →  ~12 calls/90s ep"),
    ("EMA alpha",          "0.4  (reacts in 2-3 calls)"),
    ("Threshold",          "0.40  (current run: 0.20)"),
    ("--frames 6",         "→  ~2-3s latency, 2× faster"),
]
for i, (k, v) in enumerate(rows):
    ax_info.text(stats_x, 2.35 - i*0.38, f"  {k}:", fontsize=8.5,
                 color="#566573", va="top")
    ax_info.text(stats_x + 3.2, 2.35 - i*0.38, v, fontsize=8.5,
                 color="#1C2833", va="top", fontweight="bold")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_timeline.png")
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
