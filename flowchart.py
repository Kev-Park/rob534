#!/usr/bin/env python3
"""
Flowchart of better_code.py execution.
Run: python rob534/flowchart.py
Saves: rob534/flowchart.png  (opens automatically if a viewer is available)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Color palette by module ───────────────────────────────────────────────────
M = {
    "better":   dict(face="#2E86C1", edge="#1A5276", text="white"),   # blue
    "ab_eval":  dict(face="#1E8449", edge="#145A32", text="white"),   # green
    "monitor":  dict(face="#CA6F1E", edge="#784212", text="white"),   # orange
    "lerobot":  dict(face="#7D3C98", edge="#4A235A", text="white"),   # purple
    "motor":    dict(face="#566573", edge="#2C3E50", text="white"),   # grey
    "external": dict(face="#B03A2E", edge="#7B241C", text="white"),   # red
    "decision": dict(face="#D4AC0D", edge="#9A7D0A", text="black"),   # yellow
    "io":       dict(face="#138D75", edge="#0E6655", text="white"),   # teal
    "thread":   dict(face="#E74C3C", edge="#CB4335", text="white"),   # crimson (threads)
    "process":  dict(face="#8E44AD", edge="#6C3483", text="white"),   # violet (processes)
    "start_end":dict(face="#1C2833", edge="#0D1117", text="white"),   # near-black
}

FIG_W, FIG_H = 26, 58
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("#F8F9FA")

# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def rect(x, y, w, h, label, mod, fontsize=8.5, bold=True, sub=None):
    c = M[mod]
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.12",
                       facecolor=c["face"], edgecolor=c["edge"],
                       linewidth=2, zorder=3)
    ax.add_patch(p)
    top_y = y + (0.12 if sub else 0)
    ax.text(x, top_y, label, ha="center", va="center",
            fontsize=fontsize, color=c["text"],
            fontweight="bold" if bold else "normal",
            zorder=4, multialignment="center")
    if sub:
        ax.text(x, y - h*0.22, sub, ha="center", va="center",
                fontsize=fontsize - 1.5, color=c["text"], style="italic",
                zorder=4, multialignment="center")


def oval(x, y, w, h, label, mod, fontsize=9):
    c = M[mod]
    p = mpatches.Ellipse((x, y), w, h,
                         facecolor=c["face"], edgecolor=c["edge"],
                         linewidth=2, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color=c["text"], fontweight="bold", zorder=4)


def diamond(x, y, w, h, label, mod, fontsize=8):
    c = M[mod]
    pts = np.array([[x, y+h/2], [x+w/2, y], [x, y-h/2], [x-w/2, y]])
    patch = plt.Polygon(pts, closed=True,
                        facecolor=c["face"], edgecolor=c["edge"],
                        linewidth=2, zorder=3)
    ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color=c["text"], fontweight="bold",
            zorder=4, multialignment="center")


def thread_rect(x, y, w, h, label, mod, fontsize=8, sub=None):
    c = M[mod]
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.12",
                       facecolor=c["face"], edgecolor=c["edge"],
                       linewidth=2, linestyle="--", zorder=3)
    ax.add_patch(p)
    top_y = y + (0.12 if sub else 0)
    ax.text(x, top_y, label, ha="center", va="center",
            fontsize=fontsize, color=c["text"], fontweight="bold",
            zorder=4, multialignment="center")
    if sub:
        ax.text(x, y - h*0.22, sub, ha="center", va="center",
                fontsize=fontsize - 1.5, color=c["text"], style="italic",
                zorder=4, multialignment="center")


def arr(x1, y1, x2, y2, label="", color="#444", lw=1.8, style="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                linestyle=style, connectionstyle="arc3,rad=0"),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2 + 0.15, (y1+y2)/2
        ax.text(mx, my, label, fontsize=7.5, color=color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))


def section_banner(y, label, color="#BDC3C7"):
    ax.axhline(y, color=color, lw=1, linestyle=":", zorder=1)
    ax.text(0.3, y + 0.1, label, fontsize=8, color="#7F8C8D",
            style="italic", va="bottom", zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# Layout constants
# ─────────────────────────────────────────────────────────────────────────────
CX   = 8.5   # main flow centre-x
BW   = 5.2   # standard box width
BH   = 0.72  # standard box height

TX   = 20.5  # threads/processes column x
TW   = 4.8

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    ("better_code.py", "better"),
    ("ab_eval.py",     "ab_eval"),
    ("struggle_monitor.py", "monitor"),
    ("lerobot",        "lerobot"),
    ("motor_commands.py", "motor"),
    ("Gemini / HF Hub","external"),
    ("Decision",       "decision"),
    ("CSV / Dataset I/O", "io"),
    ("Background Thread", "thread"),
    ("Subprocess / Process", "process"),
]
lx, ly = 0.5, FIG_H - 0.4
ax.text(lx, ly, "MODULE LEGEND", fontsize=9, fontweight="bold", color="#2C3E50")
ly -= 0.55
for lbl, mod in legend_items:
    c = M[mod]
    p = FancyBboxPatch((lx, ly - 0.26), 0.7, 0.42,
                       boxstyle="round,pad=0.05",
                       facecolor=c["face"], edgecolor=c["edge"], linewidth=1.5, zorder=3)
    ax.add_patch(p)
    ax.text(lx + 1.0, ly, lbl, fontsize=8, va="center", color="#2C3E50")
    ly -= 0.58

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
ax.text(CX, FIG_H - 0.5, "better_code.py — Full Execution Flowchart",
        ha="center", fontsize=14, fontweight="bold", color="#1C2833")

# ─────────────────────────────────────────────────────────────────────────────
# ── STARTUP ──────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
section_banner(FIG_H - 1.4, "STARTUP")

Y = FIG_H - 2.0
oval(CX, Y, 4.2, 0.75, "python better_code.py", "start_end", fontsize=10)

arr(CX, Y - 0.38, CX, Y - 1.05)
Y -= 1.4
rect(CX, Y, BW, BH, "argparse",
     "better", sub="--threshold  --episodes  --time  --interval  --switch-duration  --task")

arr(CX, Y - BH/2, CX, Y - 1.2)
Y -= 1.55
rect(CX, Y, BW, BH, "System Check",
     "better", sub="torch.cuda | psutil | nvidia-smi | subprocess")

arr(CX, Y - BH/2, CX, Y - 1.1)
Y -= 1.45

# Parallel: resolve_policy_path A and B
rect(CX - 1.6, Y, 2.8, BH, "resolve_policy_path()\nPolicy A  →  local .safetensors", "better", fontsize=8)
rect(CX + 1.6, Y, 2.8, BH, "resolve_policy_path()\nPolicy B  →  local .safetensors", "better", fontsize=8)
arr(CX - 1.6, Y - BH/2, CX - 0.2, Y - 1.0, color="#2E86C1")
arr(CX + 1.6, Y - BH/2, CX + 0.2, Y - 1.0, color="#2E86C1")

Y -= 1.4
rect(CX, Y, BW, BH, "do_ab_eval()", "ab_eval",
     sub="ab_eval.py — entry point", fontsize=9)

# ─────────────────────────────────────────────────────────────────────────────
# ── ONE-TIME SETUP ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
section_banner(Y - BH/2 - 0.15, "ONE-TIME SETUP (runs once before any episode)")

arr(CX, Y - BH/2, CX, Y - 1.1)
Y -= 1.45
rect(CX, Y, BW, BH, "_patch_vlm_cache()", "ab_eval",
     sub="monkey-patches SmolVLMWithExpertModel — base VLM loaded only once")

arr(CX, Y - BH/2, CX, Y - 1.0)
Y -= 1.35

# Side-by-side: _load_cfg and _open_dataset
rect(CX - 1.5, Y, 2.6, BH, "_load_cfg()\nPreTrainedConfig", "ab_eval", fontsize=8)
rect(CX + 1.5, Y, 2.6, BH, "_open_dataset()\nLeRobotDataset.create/load", "lerobot", fontsize=8)
arr(CX - 1.5, Y - BH/2, CX - 0.3, Y - 1.0, color="#1E8449")
arr(CX + 1.5, Y - BH/2, CX + 0.3, Y - 1.0, color="#7D3C98")

Y -= 1.35

# Side-by-side policy loads
rect(CX - 1.5, Y, 2.6, BH, "make_policy()\ncfg_a / cfg_b → VRAM", "lerobot", fontsize=8)
rect(CX + 1.5, Y, 2.6, BH, "make_pre_post_processors()\npre_a/b  post_a/b", "lerobot", fontsize=8)
arr(CX - 1.5, Y - BH/2, CX - 0.4, Y - 1.0, color="#7D3C98")
arr(CX + 1.5, Y - BH/2, CX + 0.4, Y - 1.0, color="#7D3C98")

Y -= 1.35
rect(CX, Y, BW, BH, "make_default_processors()", "lerobot",
     sub="teleop_action | robot_action | robot_observation")

arr(CX, Y - BH/2, CX, Y - 1.0)
Y -= 1.35
rect(CX, Y, BW, BH, "robot.connect()  +  init_keyboard_listener()", "lerobot",
     sub="_start_switch_key_listener() — q key wired to events[switch_policy]")

arr(CX, Y - BH/2, CX, Y - 1.0)
Y -= 1.35
rect(CX, Y, BW, BH, "LiveStruggleMonitor()\n+ monitor.start()", "monitor",
     sub="model=gemini-2.5-flash | interrupt_threshold | check_interval")

arr(CX, Y - BH/2, CX, Y - 0.95)
Y -= 1.3
rect(CX, Y, BW, BH, "_MonitorFeedingRobot(robot, monitor)", "ab_eval",
     sub="wraps robot — intercepts send_action() to push frames at 30 Hz")

arr(CX, Y - BH/2, CX, Y - 0.95)
Y -= 1.3

# Side-by-side: GUI process + display thread
rect(CX - 1.5, Y, 2.6, BH, "_stats_gui_process()\nmultiprocessing.Process", "process", fontsize=8)
rect(CX + 1.5, Y, 2.6, BH, "_live_display_loop()\nthreading.Thread", "thread", fontsize=8)
arr(CX - 1.5, Y - BH/2, CX - 0.3, Y - 1.0)
arr(CX + 1.5, Y - BH/2, CX + 0.3, Y - 1.0)

# Annotate thread box in side panel
thread_rect(TX, Y + 0.5, TW, 1.4,
            "_live_display_loop  [daemon thread]",
            "thread", sub="prints terminal every 1s\npushes stats_queue every 0.5s")
thread_rect(TX, Y - 1.2, TW, 1.15,
            "_stats_gui_process  [daemon process]",
            "process", sub="Tkinter EMA chart + switch markers")
arr(CX + 1.5 + 2.6/2, Y, TX - TW/2, Y, color="#E74C3C", style="--")
arr(CX - 1.5 + 2.6/2, Y, TX - TW/2, Y - 1.2, color="#8E44AD", style="--")

Y -= 1.35

# ─────────────────────────────────────────────────────────────────────────────
# ── EPISODE LOOP ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
section_banner(Y - 0.2, "EPISODE LOOP  (i = 0 … num_episodes)")

Y -= 0.65
rect(CX, Y, BW + 1.0, BH + 0.1,
     "for i in range(num_episodes)", "start_end", fontsize=9)

arr(CX, Y - BH/2, CX, Y - 1.0)
Y -= 1.35

# ── Pick policy
rect(CX, Y, BW, BH, "Select active policy", "ab_eval",
     sub="current_label = A (always, unless _held_position=True after q-switch)")

arr(CX, Y - BH/2, CX, Y - 1.0)
Y -= 1.35
rect(CX, Y, BW, BH, "monitor.reset_signal()", "monitor",
     sub="clears frame buffer, EMA → 0, episode_state reset")

arr(CX, Y - BH/2, CX, Y - 1.0)
Y -= 1.35
rect(CX, Y, BW, BH, "_go_home_with_robot(robot)", "ab_eval",
     sub="load_home() → motor_commands.py  |  robot.send_action() 30 steps")

# Side panel: motor_commands
rect(TX, Y, TW - 0.4, BH, "load_home()\nmotor_commands.py", "motor", fontsize=8)
arr(CX + BW/2, Y, TX - (TW-0.4)/2, Y, color="#566573", style="--")

arr(CX, Y - BH/2, CX, Y - 1.0)
Y -= 1.35

thread_rect(CX, Y, BW, BH, "_struggle_watcher()  [daemon thread]", "thread",
            sub="polls monitor.is_struggling() every 0.25s → sets events[auto_switched]")

# Side panel: watcher thread detail
thread_rect(TX, Y, TW, BH, "_struggle_watcher  [thread]", "thread",
            sub="every 0.25s: monitor.is_struggling()?\n→ events[exit_early] = True")
arr(CX + BW/2, Y, TX - TW/2, Y, color="#E74C3C", style="--")

arr(CX, Y - BH/2, CX, Y - 1.0)
Y -= 1.35

# ── record_loop A
rect(CX, Y, BW + 0.4, BH + 0.15, "record_loop( policy_A )", "lerobot",
     sub="30 Hz: get_observation → policy.forward → send_action → dataset buffer\n"
         "runs until control_time_s OR events[exit_early]", fontsize=8.5)

# Side panel: what happens inside record_loop
rect(TX, Y + 0.6, TW, 0.72, "_MonitorFeedingRobot.send_action()", "ab_eval", fontsize=8)
rect(TX, Y - 0.3, TW, 0.72, "monitor.push_frame()  @30 Hz", "monitor", fontsize=8)
arr(TX, Y + 0.6 - 0.36, TX, Y - 0.3 + 0.36, color="#CA6F1E")
arr(CX + (BW+0.4)/2, Y + 0.3, TX - TW/2, Y + 0.6, color="#7D3C98", style="--")

# LiveStruggleMonitor background Gemini loop
thread_rect(TX, Y - 1.3, TW, 0.82,
            "LiveStruggleMonitor._run()", "monitor",
            sub="every check_interval s: Gemini API → EMA update")
arr(TX, Y - 0.3 - 0.36, TX, Y - 1.3 + 0.41, color="#CA6F1E", style="--")
rect(TX, Y - 2.25, TW, 0.72, "Gemini  gemini-2.5-flash", "external", fontsize=8)
arr(TX, Y - 1.3 - 0.41, TX, Y - 2.25 + 0.36, color="#B03A2E")
arr(TX, Y - 2.25 - 0.36, TX, Y - 1.3 - 0.55, color="#B03A2E", label="interrupt_probability")

Y -= 1.55

arr(CX, Y - BH/2, CX, Y - 1.0)
Y -= 1.35
rect(CX, Y, BW, BH, "watcher_stop.set()\n+ monitor.get_episode_state()", "monitor",
     sub="collect ep stats: picks, drops, struggle_score, target_reached")

arr(CX, Y - BH/2, CX, Y - 0.9)
Y -= 1.2

diamond(CX, Y, 5.6, 1.0, "events[auto_switched]\n== True ?", "decision")

# ── YES path (auto_switch intervention)
arr(CX + 2.8, Y, CX + 4.2, Y, label="YES", color="#D4AC0D")
AX = CX + 4.2 + 0.1  # auto-switch column x
section_banner(Y - 0.5, "AUTO-SWITCH INTERVENTION", color="#D4AC0D")

rect(AX + 1.2, Y, 3.8, BH, "snapshot peak EMA → stats_queue\n(switch_marker + delayed_reset)", "ab_eval", fontsize=8)
arr(AX + 1.2, Y - BH/2, AX + 1.2, Y - 1.0, color="#1E8449")
Y2 = Y - 1.4
rect(AX + 1.2, Y2, 3.8, BH, "monitor.pause_for_transfer()\n+ monitor.reset_signal()", "monitor", fontsize=8)
arr(AX + 1.2, Y2 - BH/2, AX + 1.2, Y2 - 1.0, color="#7D3C98")
Y2 -= 1.35
rect(AX + 1.2, Y2, 3.8, BH + 0.15, "record_loop( policy_B )", "lerobot",
     sub="control_time_s = switch_duration\narm holds position — no go_home", fontsize=8)
arr(AX + 1.2, Y2 - BH/2, AX + 1.2, Y2 - 1.0, color="#1E8449")
Y2 -= 1.4
rect(AX + 1.2, Y2, 3.8, BH, "dataset_b.save_episode()\n+ _append_episode_stats_ab()", "io", fontsize=8)
arr(AX + 1.2, Y2 - BH/2, CX + 0.3, Y - 2.3, color="#D4AC0D")   # rejoin main flow

# ── NO path
arr(CX, Y - 1.0/2, CX, Y - 1.4, label="NO", color="#444")
Y -= 1.4

arr(CX, Y - 0.05, CX, Y - 0.95)
Y -= 1.3
rect(CX, Y, BW, BH, "sleep(2.5)  →  dataset.save_episode()", "lerobot",
     sub="wait for PNG writer threads before video encoder reads frames")

arr(CX, Y - BH/2, CX, Y - 1.0)
Y -= 1.35
rect(CX, Y, BW, BH, "_append_episode_stats_ab()", "io",
     sub="ab_eval_stats.csv — episode_index, policy, target_reached, picks, drops, EMA…")

arr(CX, Y - BH/2, CX, Y - 1.0)
Y -= 1.35

diamond(CX, Y, 5.0, 1.0, "i + 1 < num_episodes\nAND NOT stop_recording?", "decision")

# Loop back arrow
arr(CX - 2.5, Y, CX - 5.5, Y, color="#444")
ax.annotate("", xy=(CX - 5.5, Y + 18.5), xytext=(CX - 5.5, Y),
            arrowprops=dict(arrowstyle="->", color="#444", lw=1.8), zorder=2)
ax.annotate("", xy=(CX - BW/2, Y + 18.5), xytext=(CX - 5.5, Y + 18.5),
            arrowprops=dict(arrowstyle="->", color="#444", lw=1.8), zorder=2)
ax.text(CX - 5.8, Y + 9.0, "next episode", fontsize=8, color="#444",
        rotation=90, va="center")

arr(CX, Y - 1.0/2, CX, Y - 1.2, label="NO → exit loop", color="#444")
Y -= 1.6

# ─────────────────────────────────────────────────────────────────────────────
# ── FINALLY / CLEANUP ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
section_banner(Y + 0.2, "FINALLY (always runs — even on exception)")

rect(CX, Y, BW, BH, "display_stop.set()  +  monitor.stop()", "monitor",
     sub="kills _live_display_loop thread and Gemini polling thread")

arr(CX, Y - BH/2, CX, Y - 1.0)
Y -= 1.35
rect(CX, Y, BW, BH, "_go_home_with_robot(robot)", "ab_eval",
     sub="park arm safely before power-off")

arr(CX, Y - BH/2, CX, Y - 1.0)
Y -= 1.35
rect(CX, Y, BW, BH, "robot.disconnect()\n+ listener.stop() + switch_listener.stop()", "lerobot")

arr(CX, Y - BH/2, CX, Y - 1.0)
Y -= 1.35
rect(CX, Y, BW, BH, "dataset_a.finalize()  +  dataset_b.finalize()", "lerobot",
     sub="flush video writers, write index parquets, close file handles")

arr(CX, Y - BH/2, CX, Y - 0.9)
Y -= 1.2
oval(CX, Y, 3.5, 0.72, "END", "start_end", fontsize=10)

# ─────────────────────────────────────────────────────────────────────────────
# ── SIDE PANEL HEADER ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
ax.text(TX, FIG_H - 1.8, "BACKGROUND THREADS / PROCESSES",
        ha="center", fontsize=9, fontweight="bold", color="#2C3E50",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ECF0F1", ec="#BDC3C7"))
ax.axvline(TX - TW/2 - 0.3, color="#BDC3C7", lw=1.2, linestyle=":", zorder=1)
ax.text(TX - TW/2 - 0.5, FIG_H/2, "dashed border = daemon thread/process",
        fontsize=7.5, color="#95A5A6", rotation=90, va="center")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flowchart.png")
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
