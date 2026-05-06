"""
================================================================================
HOW TO USE ab_eval.py
================================================================================

WHAT IT DOES
------------
Runs two policies back to back on the same robot. Starts on policy A and stays
there until you manually press 'q' to switch. You control when the switch
happens — there is no automatic alternation. Data is saved to two separate
HuggingFace datasets so you can compare results later.

SETUP
-----
1. Plug the arm into COM5
2. Plug the camera in (USB, camera index 1)
3. Make sure you are logged into HuggingFace:
       huggingface-cli login
4. Edit the call at the bottom of better_code.py (or your own script):

       from ab_eval import do_ab_eval
       from better_code import resolve_policy_path

       do_ab_eval(
           policy_path_a=resolve_policy_path("SkywalkerLi/smolvla-phase-split"),
           policy_path_b=resolve_policy_path("SkywalkerLi/smolvla-aug"),
           repo_id_a="SkywalkerLi/eval_smolvla-phase-split",
           repo_id_b="SkywalkerLi/eval_smolvla-aug",
           num_episodes=20,       # total episodes across both policies
           episode_time_s=45,     # max seconds per episode
       )

   Change the model names and repo IDs to whatever you are testing.
   resolve_policy_path() checks if the model is already downloaded locally
   and skips the download if so.

STARTUP
-------
When you run it, both policies load into GPU memory before anything moves.
This takes about 20-30 seconds each. It only happens once — no reloading
between episodes.

KEYBOARD CONTROLS (while an episode is running)
-----------------------------------------------
    q        Switch to the other policy.
             - Ends the current episode immediately
             - Motors FREEZE at their exact position (no go_home)
             - The episode data is saved to BOTH datasets
             - Next episode starts with the other policy from the same position
             Use this when you want a direct comparison with the scene unchanged.

    d / →    End the episode early the normal way.
             - Arm goes back to home position before the next episode
             - Data saved only to the current policy's dataset
             - Same policy continues next episode

    Escape   Stop the whole run cleanly.
             - Arm goes home, datasets are finalized, robot disconnects

WHAT HAPPENS WHEN YOU PRESS q
------------------------------
Say policy A is running and you press q at 12 seconds:
  - Episode ends, 360 frames are in the buffer
  - Those 360 frames get copied into policy B's dataset too
  - Both datasets save the episode
  - Motors stay frozen — cube is still in the same spot
  - Next episode: policy B takes over from that exact position
  - Press q again to switch back to A

NORMAL EPISODE (no q press)
----------------------------
  - Episode runs until time limit (episode_time_s) or you press d
  - Arm goes home between episodes
  - Data saved only to the active policy's dataset
  - Same policy runs again next episode

WHERE DATA IS SAVED
-------------------
Locally at:
    C:\\Users\\<you>\\.cache\\huggingface\\lerobot\\<repo_id>\\

If you stop and restart, the script picks up where it left off automatically
(it checks for meta/tasks.parquet to detect a valid existing dataset).

TO UPLOAD AFTER YOU ARE DONE
-----------------------------
    python push_eval_to_hub.py <local_path> <hub_repo_id>

    Example:
    python push_eval_to_hub.py C:/Users/calle/.cache/huggingface/lerobot/SkywalkerLi/eval_smolvla-phase-split SkywalkerLi/eval_smolvla-phase-split

================================================================================
"""

import shutil
import subprocess
import threading
import time as _time
from pathlib import Path

# PIL sometimes raises "image file is truncated" when the background image-writer
# threads haven't finished flushing a PNG before the video encoder reads it.
# LOAD_TRUNCATED_IMAGES tells PIL to load whatever data is present instead of
# crashing — the truncation is typically just the last few bytes of a frame.
import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

# Helpers imported from better_code.py (must be in the same directory):
#   _check_starvation     — warns if CPU/RAM is under pressure before opening camera
#   _go_home_with_robot   — smoothly moves arm to the position saved in home_pos.json
#   _rerun_is_running     — checks port 9090 so we don't restart rerun if already open
#   _wait_for_port        — blocks until a TCP port is accepting connections
#   _wait_and_open_viewer — waits for rerun server to start then opens the browser tab
#   repo_id_from_policy   — derives "SkywalkerLi/eval_<model>" from a policy path
from better_code import (
    _check_starvation,
    _go_home_with_robot,
    _rerun_is_running,
    _wait_and_open_viewer,
    _wait_for_port,
    repo_id_from_policy,
)


def _append_episode_stats_ab(csv_path: str, ep_idx: int, policy_label: str, ep_state) -> None:
    """Append one episode's stats row to a CSV (creates file + header on first write).

    Columns: episode_index, policy, then all EpisodeState dataclass fields.
    """
    import csv
    from dataclasses import asdict
    path = Path(csv_path)
    row = {"episode_index": ep_idx, "policy": policy_label, **asdict(ep_state)}
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  [stats] ep {ep_idx} (Policy {policy_label}) -> {csv_path}")


def _duplicate_buffer(source_dataset, target_dataset) -> int:
    """
    Copy every buffered frame from source_dataset into target_dataset.

    Called when 'q' is pressed so both datasets receive the episode that was
    mid-flight when the switch happened.

    IMPORTANT: must be called BEFORE source_dataset.save_episode() because
    save_episode() clears the buffer — calling this after would copy nothing.

    How it works:
      episode_buffer is a dict where each feature key (e.g. "action",
      "observation.state", "observation.images.camera1") maps to a list of
      per-frame values. We iterate frame-by-frame and call add_frame() on
      the target, which lets that dataset manage its own index counters.
      Metadata keys (frame_index, episode_index, timestamp, etc.) are skipped
      because add_frame() recomputes them — passing the source values in would
      cause index collisions between the two datasets.

    Returns number of frames copied (0 if the buffer was empty).
    """
    buf = source_dataset.episode_buffer
    if buf is None or buf.get("size", 0) == 0:
        return 0

    n_frames = buf["size"]

    # Skip keys that the dataset manages internally
    skip_keys = {"size", "frame_index", "episode_index", "index", "task_index", "timestamp"}
    data_keys = [k for k in buf if k not in skip_keys]

    import numpy as np
    from PIL import Image as PILImage

    for i in range(n_frames):
        frame = {}
        for k in data_keys:
            val = buf[k]
            if hasattr(val, "__len__") and i < len(val):
                v = val[i]
                # VideoEncodingManager stores image frames as file paths (strings).
                # Reload the actual pixel data before passing to the target dataset.
                # Only do this for image keys — other string values (e.g. "task") pass through as-is.
                if isinstance(v, str) and "image" in k:
                    v = np.array(PILImage.open(v))
                frame[k] = v
        if frame:
            target_dataset.add_frame(frame)

    return n_frames


def _start_switch_key_listener(events):
    """
    Starts a second background keyboard listener that watches only for 'q'.

    lerobot's init_keyboard_listener() already handles d/→ (exit early),
    a/← (rerecord), and Escape (stop recording). We add 'q' as a separate
    pynput listener on top — pynput supports multiple concurrent listeners
    on the same keyboard without conflict.

    When 'q' is pressed:
      - events["exit_early"] = True   → record_loop exits on its next iteration
      - events["switch_policy"] = True → do_ab_eval sees this after the episode
                                         and skips go_home (motors hold position)
                                         then flips current_label to the other policy

    Returns the started listener so the caller can stop it in a finally block.
    """
    from pynput import keyboard

    def on_press(key):
        try:
            if key.char == "q":
                events["exit_early"] = True
                events["switch_policy"] = True
                print("\n  [q] Policy switch — holding position, ending episode early.")
        except AttributeError:
            pass  # special keys (shift, ctrl, arrows) don't have .char

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def _stats_gui_process(queue, score_mode: str = "max") -> None:
    """Standalone tkinter stats window — runs in its own process, fully isolated.

    Receives dicts from the main process via a multiprocessing.Queue and
    refreshes labels every 500 ms.  Completely separate from the robot
    control loop so it cannot cause GIL contention or slow down switching.
    """
    import tkinter as tk
    import collections

    BG   = "#111111"
    FG   = "#dddddd"
    GREY = "#666666"

    # ── chart constants ───────────────────────────────────────────────────────
    CW        = 292          # chart canvas width  (pixels)
    CH        = 120          # chart canvas height (pixels)
    MAX_PTS   = 120          # rolling history length (~2 min at 1 pt/s)
    PAD_LEFT  = 28           # space for y-axis labels
    PAD_RIGHT = 4
    PAD_TOP   = 6
    PAD_BOT   = 4
    PLOT_W    = CW - PAD_LEFT - PAD_RIGHT
    PLOT_H    = CH - PAD_TOP  - PAD_BOT

    root = tk.Tk()
    root.title("A/B Eval — Live Stats")
    root.configure(bg=BG)
    root.geometry("400x620")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    def _row(parent, label, default="--", big=False):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill=tk.X, padx=14, pady=2)
        tk.Label(f, text=label, bg=BG, fg=GREY, width=16, anchor="w",
                 font=("Consolas", 9)).pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        size = 14 if big else 10
        lbl = tk.Label(f, textvariable=var, bg=BG, fg=FG, anchor="w",
                       font=("Consolas", size, "bold"))
        lbl.pack(side=tk.LEFT)
        return var, lbl

    tk.Label(root, text="A/B EVAL", bg=BG, fg="#aaaaaa",
             font=("Consolas", 10)).pack(pady=(10, 2))

    v_policy,  l_policy  = _row(root, "policy",        big=True)
    v_timer,   _         = _row(root, "episode time",   big=True)
    v_metric,  _         = _row(root, "S score")
    v_prob,    _         = _row(root, "interrupt prob")
    v_status,  l_status  = _row(root, "status")

    # ── per-judge vote boxes ───────────────────────────────────────────────────
    votes_frame = tk.Frame(root, bg=BG)
    votes_frame.pack(fill=tk.X, padx=14, pady=(6, 2))
    tk.Label(votes_frame, text="judge votes", bg=BG, fg=GREY,
             font=("Consolas", 8)).pack(anchor="w")

    vote_box_frame = tk.Frame(votes_frame, bg=BG)
    vote_box_frame.pack(anchor="w", pady=(2, 0))
    # Pre-create 5 vote box labels (one per judge temperature) showing temp + confidence
    _vote_labels: list[tk.Label] = []
    for _ in range(5):
        lbl = tk.Label(vote_box_frame, text="--\n--", width=6,
                       bg="#333333", fg="#999999",
                       font=("Consolas", 8, "bold"), relief="flat", padx=4, pady=2,
                       justify="center")
        lbl.pack(side=tk.LEFT, padx=2)
        _vote_labels.append(lbl)

    def _update_vote_boxes(votes):
        """Update the vote box labels — shows temperature and confidence per judge."""
        for i, lbl in enumerate(_vote_labels):
            if i < len(votes):
                v = votes[i]
                struggling = v.get("struggling", False)
                temp = v.get("temp", 0.0)
                conf = v.get("confidence", 0.0)
                lbl.config(
                    text=f"{temp:.2f}\n{conf:.2f}",
                    bg="#cc2222" if struggling else "#22aa22",
                    fg="#ffffff",
                )
            else:
                lbl.config(text="--\n--", bg="#333333", fg="#999999")

    # ── per-channel raw values + ratio bars ──────────────────────────────────
    _CH_NAMES  = ("E_RMS", "J_RMS", "neg_SPARC", "rho_HF", "sigma_bar")
    _CH_LABELS = ("E_rms", "Jerk",  "SPARC",     "HF_pwr", "Stall")
    _CH_BAR_MAX = 2.0   # ratio value that fills the bar to 100%

    ch_frame = tk.Frame(root, bg=BG)
    ch_frame.pack(fill=tk.X, padx=14, pady=(6, 2))
    tk.Label(ch_frame, text="channels  raw · ratio/θ", bg=BG, fg=GREY,
             font=("Consolas", 8)).pack(anchor="w")

    # Each row: label | raw_val_lbl | bar | ratio_lbl
    _ch_rows: list[tuple] = []   # (raw_var, bar_cv, bar_id, ratio_var, theta_var)
    _CH_BAR_W, _CH_BAR_H = 120, 10
    for ch_label in _CH_LABELS:
        row = tk.Frame(ch_frame, bg=BG)
        row.pack(fill=tk.X, pady=1)
        tk.Label(row, text=f"{ch_label:<9}", bg=BG, fg=GREY,
                 font=("Consolas", 8), width=9, anchor="w").pack(side=tk.LEFT)
        raw_var = tk.StringVar(value="--")
        tk.Label(row, textvariable=raw_var, bg=BG, fg="#dddddd",
                 font=("Consolas", 8), width=9, anchor="e").pack(side=tk.LEFT)
        bar_cv = tk.Canvas(row, width=_CH_BAR_W, height=_CH_BAR_H,
                           bg="#1a1a1a", highlightthickness=0)
        bar_cv.pack(side=tk.LEFT, padx=(4, 4))
        bar_cv.create_rectangle(0, 0, _CH_BAR_W, _CH_BAR_H,
                                fill="#222222", outline="")
        bar_id = bar_cv.create_rectangle(0, 0, 0, _CH_BAR_H, fill="#00e676", outline="")
        ratio_var = tk.StringVar(value="")
        tk.Label(row, textvariable=ratio_var, bg=BG, fg=GREY,
                 font=("Consolas", 8), width=6, anchor="w").pack(side=tk.LEFT)
        theta_var = tk.StringVar(value="")
        tk.Label(row, textvariable=theta_var, bg=BG, fg="#666688",
                 font=("Consolas", 8), width=8, anchor="w").pack(side=tk.LEFT)
        _ch_rows.append((raw_var, bar_cv, bar_id, ratio_var, theta_var))

    # S score row (prominent, just below channels)
    s_frame = tk.Frame(root, bg=BG)
    s_frame.pack(fill=tk.X, padx=14, pady=(4, 2))
    tk.Label(s_frame, text=f"S = {score_mode}(ratio)", bg=BG, fg=GREY,
             font=("Consolas", 8)).pack(side=tk.LEFT)
    v_s_score = tk.StringVar(value="--")
    lbl_s_score = tk.Label(s_frame, textvariable=v_s_score, bg=BG, fg="#dddddd",
                           font=("Consolas", 10, "bold"))
    lbl_s_score.pack(side=tk.LEFT, padx=(8, 0))

    def _update_channel_bars(ratios: dict, values: dict, thetas: dict | None = None, s_val: float | None = None):
        thr = _chart_state["thr"]
        dominant = max(ratios, key=ratios.get) if ratios else None
        for i, ch_name in enumerate(_CH_NAMES):
            raw_var, bar_cv, bar_id, ratio_var, theta_var = _ch_rows[i]
            raw = values.get(ch_name, float("nan"))
            ratio = ratios.get(ch_name, 0.0)
            if ratio != ratio:  # NaN guard
                ratio = 0.0
            # raw value label — scientific notation keeps it compact
            raw_var.set(f"{raw:.4g}" if raw == raw else "--")  # nan check
            # ratio bar
            filled = max(0, int(min(ratio, _CH_BAR_MAX) / _CH_BAR_MAX * _CH_BAR_W))
            is_over = ratio >= thr
            color = "#ff3333" if is_over else "#00e676"
            bar_cv.coords(bar_id, 0, 0, filled, _CH_BAR_H)
            bar_cv.itemconfig(bar_id, fill=color)
            ratio_var.set(f"r={ratio:.2f}")
            # theta (calibration threshold used for normalization)
            if thetas:
                theta = thetas.get(ch_name, float("nan"))
                theta_var.set(f"θ={theta:.4g}" if theta == theta else "θ=--")
            else:
                theta_var.set("")
        # S score line
        if ratios:
            if s_val is None:
                s_val = max(ratios.values())
            v_s_score.set(f"{s_val:.3f}  ({dominant})")
            lbl_s_score.config(fg="#ff3333" if s_val >= thr else "#44ff44")
        else:
            v_s_score.set("--")
            lbl_s_score.config(fg=GREY)

    # ── live S score chart ────────────────────────────────────────────────────
    chart_frame = tk.Frame(root, bg=BG)
    chart_frame.pack(fill=tk.X, padx=10, pady=(8, 6))
    tk.Label(chart_frame, text="S score  (─ threshold)", bg=BG, fg=GREY,
             font=("Consolas", 8)).pack(anchor="w")

    cv = tk.Canvas(chart_frame, width=CW, height=CH,
                   bg="#0d1117", highlightthickness=1, highlightbackground="#333333")
    cv.pack()

    def _ema_to_y(v):
        """Map S value [0, 2] → canvas y (top=2, bottom=0)."""
        return PAD_TOP + int((1.0 - min(max(v, 0.0), 2.0) / 2.0) * PLOT_H)

    def _idx_to_x(i, n):
        """Map history index i (out of n points) → canvas x."""
        if n <= 1:
            return PAD_LEFT + PLOT_W
        return PAD_LEFT + int(i * PLOT_W / (n - 1))

    # Static y-axis labels (0.0, 1.0, 2.0)
    for val, label_txt in [(0.0, "0.0"), (1.0, "1.0"), (2.0, "2.0")]:
        y = _ema_to_y(val)
        cv.create_line(PAD_LEFT, y, PAD_LEFT + PLOT_W, y,
                       fill="#1e2530", width=1)          # faint grid line
        cv.create_text(PAD_LEFT - 4, y, text=label_txt,
                       anchor="e", fill=GREY, font=("Consolas", 7))

    # Threshold line — repositioned on each update
    thr_line  = cv.create_line(PAD_LEFT, 60, PAD_LEFT + PLOT_W, 60,
                               fill="#ff8800", width=1, dash=(6, 4))
    thr_label = cv.create_text(PAD_LEFT + PLOT_W - 2, 60,
                               text="", anchor="se",
                               fill="#ff8800", font=("Consolas", 7, "bold"))

    # EMA history and switch marker tracking
    ema_history   = collections.deque(maxlen=MAX_PTS)
    label_history = collections.deque(maxlen=MAX_PTS)   # "A" or "B" per point
    _chart_state  = {"thr": 0.6, "counter": 0}
    switch_markers: list[int] = []   # data-point counter values at each switch

    _POLICY_COLOR = {"A": "#00ffff", "B": "#ff44ff"}   # cyan / magenta

    def _redraw_chart(ema, thr, label="A"):
        _chart_state["thr"] = thr
        ema_history.append(ema)
        label_history.append(label)
        _chart_state["counter"] += 1
        n = len(ema_history)

        # Reposition threshold line
        ty = _ema_to_y(thr)
        cv.coords(thr_line, PAD_LEFT, ty, PAD_LEFT + PLOT_W, ty)
        cv.itemconfig(thr_label, text=f"{thr:.2f}", anchor="ne")
        cv.coords(thr_label, PAD_LEFT + PLOT_W - 2, ty - 2)

        # Remove previous EMA drawing
        cv.delete("ema_plot")

        if n < 2:
            return

        # ── Switch marker vertical lines (drawn first, EMA renders on top) ──
        for sc in switch_markers:
            offset = _chart_state["counter"] - sc
            idx    = n - 1 - offset
            if 0 <= idx < n:
                sx = _idx_to_x(idx, n)
                cv.create_line(sx, PAD_TOP, sx, PAD_TOP + PLOT_H,
                               fill="#ffdd00", width=1, dash=(3, 3),
                               tags="ema_plot")
                cv.create_text(sx + 2, PAD_TOP + 1, text="switch",
                               anchor="nw", fill="#ffdd00",
                               font=("Consolas", 6, "bold"), tags="ema_plot")

        # Build point list
        pts = []
        for i, v in enumerate(ema_history):
            pts.append(_idx_to_x(i, n))
            pts.append(_ema_to_y(v))

        # Shade danger zone (between threshold line and EMA line when above thr)
        # Build a filled polygon: EMA line → right → bottom-right → bottom-left
        above = [v for v in ema_history if v >= thr]
        if above:
            poly_pts = []
            for i, v in enumerate(ema_history):
                poly_pts.append(_idx_to_x(i, n))
                poly_pts.append(min(_ema_to_y(v), ty))   # clamp to threshold
            # close polygon at threshold level
            poly_pts += [_idx_to_x(n - 1, n), ty, _idx_to_x(0, n), ty]
            cv.create_polygon(poly_pts, fill="#3a0000", outline="",
                              tags="ema_plot")

        # Draw EMA line segmented by policy — cyan (A) / magenta (B).
        # Consecutive same-label points form one create_line call; the
        # transition point is shared between segments for a clean join.
        seg_pts:   list[float] = []
        seg_label: str | None  = None
        for i, (v, lbl) in enumerate(zip(ema_history, label_history)):
            x, y = _idx_to_x(i, n), _ema_to_y(v)
            if lbl != seg_label:
                if len(seg_pts) >= 4:
                    cv.create_line(*seg_pts,
                                   fill=_POLICY_COLOR.get(seg_label, "#00ffff"),
                                   width=2, smooth=True, tags="ema_plot")
                # overlap one point at the boundary for visual continuity
                seg_pts   = ([seg_pts[-2], seg_pts[-1]] if seg_pts else []) + [x, y]
                seg_label = lbl
            else:
                seg_pts += [x, y]
        if len(seg_pts) >= 4:
            cv.create_line(*seg_pts,
                           fill=_POLICY_COLOR.get(seg_label, "#00ffff"),
                           width=2, smooth=True, tags="ema_plot")

        line_color = _POLICY_COLOR.get(label, "#00ffff")

        # Dot at current value (rightmost point)
        cx = _idx_to_x(n - 1, n)
        cy = _ema_to_y(ema)
        cv.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                       fill=line_color, outline="", tags="ema_plot")

        # Current value text
        cv.create_text(cx + 5, cy, text=f"{ema:.3f}", anchor="w",
                       fill=line_color, font=("Consolas", 7, "bold"),
                       tags="ema_plot")

    _redraw_chart(0.0, 0.6)

    # ── Per-channel metrics window ─────────────────────────────────────────────
    _CHANNELS = ("E_RMS", "J_RMS", "neg_SPARC", "rho_HF", "sigma_bar")
    MCW       = 310   # mini-chart canvas width
    MCH       = 58    # mini-chart canvas height
    MC_PAD_L  = 26
    MC_PAD_R  = 4
    MC_PAD_T  = 4
    MC_PAD_B  = 4
    MC_PLOT_W = MCW - MC_PAD_L - MC_PAD_R
    MC_PLOT_H = MCH - MC_PAD_T  - MC_PAD_B
    MC_Y_MAX  = 1.25  # y-axis ceiling (ratio is [0,1]; extra headroom for spikes)

    metrics_win = tk.Toplevel(root)
    metrics_win.title("Channel Metrics")
    metrics_win.configure(bg=BG)
    metrics_win.geometry("360x470+415+0")
    metrics_win.resizable(False, False)
    metrics_win.attributes("-topmost", True)

    tk.Label(metrics_win, text="CHANNEL METRICS  (m / θ)",
             bg=BG, fg="#aaaaaa", font=("Consolas", 9)).pack(pady=(8, 2))

    ch_histories:       dict = {ch: collections.deque(maxlen=MAX_PTS) for ch in _CHANNELS}
    ch_label_histories: dict = {ch: collections.deque(maxlen=MAX_PTS) for ch in _CHANNELS}
    ch_canvases:        dict = {}
    ch_val_vars:        dict = {}

    def _ratio_to_y(r):
        frac = max(0.0, min(r / MC_Y_MAX, 1.0))
        return MC_PAD_T + (1.0 - frac) * MC_PLOT_H

    def _mc_x(i, n):
        return MC_PAD_L + (i / max(n - 1, 1)) * MC_PLOT_W

    for _ch in _CHANNELS:
        _fr = tk.Frame(metrics_win, bg=BG)
        _fr.pack(fill=tk.X, padx=10, pady=(3, 0))
        _hdr = tk.Frame(_fr, bg=BG)
        _hdr.pack(fill=tk.X)
        tk.Label(_hdr, text=_ch, bg=BG, fg="#aaaaaa",
                 font=("Consolas", 8, "bold"), width=11, anchor="w").pack(side=tk.LEFT)
        _vv = tk.StringVar(value="--")
        ch_val_vars[_ch] = _vv
        tk.Label(_hdr, textvariable=_vv, bg=BG, fg=FG,
                 font=("Consolas", 8)).pack(side=tk.LEFT)
        _cv = tk.Canvas(_fr, width=MCW, height=MCH,
                        bg="#0d1117", highlightthickness=1,
                        highlightbackground="#333333")
        _cv.pack()
        ch_canvases[_ch] = _cv
        # Y-axis labels and grid
        for _v, _t in ((0.0, "0"), (0.5, ".5"), (1.0, "1")):
            _yy = _ratio_to_y(_v)
            _cv.create_text(MC_PAD_L - 3, _yy, text=_t, anchor="e",
                            fill="#555566", font=("Consolas", 6))
        for _v in (0.25, 0.5, 0.75):
            _cv.create_line(MC_PAD_L, _ratio_to_y(_v),
                            MC_PAD_L + MC_PLOT_W, _ratio_to_y(_v),
                            fill="#1a2030", width=1)
        # P95 threshold line at ratio = 1.0
        _y1 = _ratio_to_y(1.0)
        _cv.create_line(MC_PAD_L, _y1, MC_PAD_L + MC_PLOT_W, _y1,
                        fill="#ff8800", width=1, dash=(4, 3))

    def _redraw_channels(ratios: dict, label: str):
        counter = _chart_state["counter"]
        for ch in _CHANNELS:
            ratio = ratios.get(ch, 0.0)
            ch_histories[ch].append(ratio)
            ch_label_histories[ch].append(label)
            ch_val_vars[ch].set(f"{ratio:.3f}")
            cv_ch = ch_canvases[ch]
            cv_ch.delete("mc_plot")
            hist  = ch_histories[ch]
            lhist = ch_label_histories[ch]
            n = len(hist)
            if n < 2:
                continue
            # Segmented line by policy color
            seg_pts:   list = []
            seg_lbl:   str | None = None
            for i, (v, lbl) in enumerate(zip(hist, lhist)):
                x, y = _mc_x(i, n), _ratio_to_y(v)
                if lbl != seg_lbl:
                    if len(seg_pts) >= 4:
                        cv_ch.create_line(*seg_pts,
                                          fill=_POLICY_COLOR.get(seg_lbl, "#00ffff"),
                                          width=1, smooth=True, tags="mc_plot")
                    seg_pts = ([seg_pts[-2], seg_pts[-1]] if seg_pts else []) + [x, y]
                    seg_lbl = lbl
                else:
                    seg_pts += [x, y]
            if len(seg_pts) >= 4:
                cv_ch.create_line(*seg_pts,
                                  fill=_POLICY_COLOR.get(seg_lbl, "#00ffff"),
                                  width=1, smooth=True, tags="mc_plot")
            # Tip dot
            cx, cy = _mc_x(n - 1, n), _ratio_to_y(ratio)
            cv_ch.create_oval(cx - 2, cy - 2, cx + 2, cy + 2,
                              fill=_POLICY_COLOR.get(label, "#00ffff"),
                              outline="", tags="mc_plot")
            # Switch markers (shared with main chart via switch_markers list)
            for sc in switch_markers:
                idx = n - 1 - (counter - sc)
                if 0 <= idx < n:
                    sx = _mc_x(idx, n)
                    cv_ch.create_line(sx, MC_PAD_T, sx, MC_PAD_T + MC_PLOT_H,
                                      fill="#ffdd00", width=1, dash=(2, 2),
                                      tags="mc_plot")

    def _reset_display():
        """Clear chart history, switch markers, and all text widgets."""
        ema_history.clear()
        label_history.clear()
        switch_markers.clear()
        for ch in _CHANNELS:
            ch_histories[ch].clear()
            ch_label_histories[ch].clear()
            ch_canvases[ch].delete("mc_plot")
            ch_val_vars[ch].set("--")
        _chart_state["counter"] = 0
        cv.delete("ema_plot")
        v_timer.set("0:00")
        v_metric.set("0.000")
        v_prob.set("0.000")
        _update_vote_boxes([])
        _update_channel_bars({}, {}, {})
        v_status.set("--")
        l_status.config(fg=FG)
        _redraw_chart(0.0, _chart_state["thr"])

    # Delayed-reset support: keep chart visible for a moment after a switch
    # so the user can see the marker before the chart clears.
    _pending_reset = [None]

    def _cancel_pending_reset():
        if _pending_reset[0] is not None:
            root.after_cancel(_pending_reset[0])
            _pending_reset[0] = None

    def _schedule_reset(delay_ms=0):
        _cancel_pending_reset()
        if delay_ms > 0:
            _pending_reset[0] = root.after(delay_ms, _reset_display)
        else:
            _reset_display()

    def poll():
        import queue as _q
        while True:
            try:
                data = queue.get_nowait()
            except _q.Empty:
                break
            try:
                if data.get("reset"):
                    _schedule_reset(data.get("delay_ms", 0))
                    continue
                if data.get("switch_marker"):
                    # Record the current counter so we can compute the x-position
                    # on subsequent redraws as the history scrolls.
                    switch_markers.append(_chart_state["counter"])
                    continue
                label = data.get("label", "A")
                color = "#00ffff" if label == "A" else "#ff44ff"
                v_policy.set("STUDENT (A)" if label == "A" else "TEACHER (B)")
                l_policy.config(fg=color)

                ema = data.get("ema", 0.0)
                thr = data.get("threshold", 0.6)
                struggling = ema >= thr
                _redraw_chart(ema, thr, label)
                _redraw_channels(data.get("ratios", {}), label)
                ep_s = data.get("ep_elapsed", 0.0)
                mins, secs = divmod(int(ep_s), 60)
                v_timer.set(f"{mins}:{secs:02d}")
                v_metric.set(f"{data.get('ema', 0.0):.3f}")
                v_prob.set(f"{data.get('prob', 0.0):.3f}")
                _update_vote_boxes(data.get("votes", []))
                _update_channel_bars(data.get("ratios", {}), data.get("values", {}), data.get("thetas", {}), s_val=data.get("ema"))
                if struggling:
                    v_status.set("STRUGGLING")
                    l_status.config(fg="#ff2222")
                else:
                    v_status.set("ok")
                    l_status.config(fg="#44ff44")
            except Exception:
                import traceback; traceback.print_exc()
        root.after(200, poll)

    root.after(200, poll)
    root.mainloop()


def _live_display_loop(
    monitor,
    label_ref: list,
    stop_evt: threading.Event,
    interrupt_threshold: float = 0.5,
    stats_queue=None,           # multiprocessing.Queue to the GUI process
    window_name: str = "A/B Live Eval",
) -> None:
    """Print live stats to terminal every second; push to GUI every 0.5 s."""
    _gui_interval   = 0.5   # seconds between GUI pushes
    _print_interval = 1.0   # seconds between terminal prints
    _last_gui_push  = 0.0
    _last_print     = 0.0
    # Last non-empty channel data — kept across resets so the GUI doesn't go
    # blank during the brief window after a policy switch clears the monitor.
    _cached_ratios: dict = {}
    _cached_values: dict = {}
    _cached_thetas: dict = {}

    while not stop_evt.is_set():
        now     = _time.monotonic()
        intr    = monitor.get_interrupt()
        ema     = monitor.get_struggle_score()
        import math as _math
        if _math.isnan(ema) or _math.isinf(ema):
            ema = 0.0
        _r = monitor.get_channel_ratios()
        _v = monitor.get_channel_values()
        _t = monitor.get_channel_thetas()
        if _r:
            _cached_ratios = _r
        if _v:
            _cached_values = _v
        if _t:
            _cached_thetas = _t
        ratios = _cached_ratios
        values = _cached_values
        thetas = _cached_thetas
        prob    = intr.get("interrupt_probability", 0.0)
        votes   = intr.get("votes", [])
        label   = label_ref[0]
        status  = "STRUGGLING" if ema >= interrupt_threshold else "ok"

        # Print a new line to terminal every _print_interval seconds
        if now - _last_print >= _print_interval:
            t = _time.strftime("%H:%M:%S")
            bar_filled = int(ema / max(interrupt_threshold, 1e-6) * 20)
            bar = "#" * min(bar_filled, 20) + "-" * max(20 - bar_filled, 0)
            buf   = monitor.buf_len
            age   = monitor.secs_since_last_check
            age_s = f"{age:.0f}s ago" if age > 0 else "no check yet"
            vote_str = " ".join(
                f"{'Y' if v['struggling'] else 'N'}@{v['temp']:.2f}"
                for v in votes
            ) if votes else "--"
            ch_str = "  ".join(
                f"{k}={values.get(k, float('nan')):.3g}(r={v:.2f})"
                for k, v in ratios.items()
            ) if ratios else "no data"
            print(
                f"[{t}] policy={label}  [{bar}] S={ema:.3f}/{interrupt_threshold:.2f}"
                f"  gemini_p={prob:.3f}  [{vote_str}]"
                f"  buf={buf}fr  last_check={age_s}  {status}",
                flush=True,
            )
            print(f"         channels: {ch_str}", flush=True)
            _last_print = now

        # Push to GUI at its own rate
        if stats_queue is not None and not monitor.transfer_active and (now - _last_gui_push) >= _gui_interval:
            try:
                stats_queue.put_nowait({
                    "label":      label,
                    "ema":        ema,
                    "threshold":  interrupt_threshold,
                    "prob":       prob,
                    "votes":      votes,
                    "ep_elapsed": monitor.episode_elapsed,
                    "ratios":     ratios,
                    "values":     values,
                    "thetas":     thetas,
                })
                _last_gui_push = now
            except Exception:
                pass   # queue full — GUI thread hasn't drained yet, skip

        stop_evt.wait(0.1)


class _MonitorFeedingRobot:
    """Thin robot wrapper that feeds camera frames and joint state to the struggle monitor.

    Intercepts send_action() (called at ~30 Hz) to push the camera's
    latest_frame and the most recent joint state to the monitor at action rate.

    All other attributes delegate transparently to the real robot.
    """

    _CAM_NAME   = "camera1"
    _CAM_KEY    = "observation.images.camera1"
    # Joint keys in a fixed order — used to build state and action arrays
    _JOINT_KEYS = (
        "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
        "wrist_flex.pos",   "wrist_roll.pos",    "gripper.pos",
    )

    def __init__(self, robot, monitor):
        object.__setattr__(self, "_robot",       robot)
        object.__setattr__(self, "_monitor",     monitor)
        object.__setattr__(self, "_last_state",  None)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_robot"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_robot"), name, value)

    def _push_latest_camera_frame(self):
        """Push the camera's latest background-thread frame to the monitor."""
        import numpy as np
        robot   = object.__getattribute__(self, "_robot")
        monitor = object.__getattribute__(self, "_monitor")
        try:
            cam = robot.cameras.get(self._CAM_NAME)
            if cam is not None and cam.latest_frame is not None:
                frame = cam.latest_frame
                if not isinstance(frame, np.ndarray):
                    frame = np.array(frame)
                monitor.push_frame(np.ascontiguousarray(frame))
        except Exception:
            pass

    def send_action(self, action):
        """Delegate to real robot and push camera frame + joint obs to monitor at ~30 Hz."""
        import numpy as np
        robot   = object.__getattribute__(self, "_robot")
        monitor = object.__getattribute__(self, "_monitor")
        self._push_latest_camera_frame()
        last_state = object.__getattribute__(self, "_last_state")
        if last_state is not None:
            # Extract action array from dict (joint keys) or use directly if array
            if isinstance(action, dict):
                action_vals = [float(action[k]) for k in self._JOINT_KEYS if k in action]
                action_arr  = np.array(action_vals, dtype=np.float32)
            else:
                action_arr = np.asarray(action, dtype=np.float32).flatten()
            monitor.push_observation(action_arr, last_state)
        return robot.send_action(action)

    def get_observation(self):
        import numpy as np
        robot   = object.__getattribute__(self, "_robot")
        monitor = object.__getattribute__(self, "_monitor")
        obs = robot.get_observation()
        # Build state vector from individual joint-position keys
        state_vals = [float(obs[k]) for k in self._JOINT_KEYS if k in obs]
        if state_vals:
            object.__setattr__(self, "_last_state",
                               np.array(state_vals, dtype=np.float32))
        frame = obs.get(self._CAM_KEY)
        if frame is not None:
            # lerobot returns PIL Images; convert to BGR numpy for the monitor
            if hasattr(frame, "numpy"):          # tensor
                frame = frame.numpy()
            if hasattr(frame, "convert"):        # PIL Image
                import numpy as np
                frame = np.array(frame.convert("RGB"))[:, :, ::-1]
            elif isinstance(frame, np.ndarray) and frame.ndim == 3:
                pass                             # already HWC numpy
            monitor.push_frame(np.ascontiguousarray(frame))
        return obs


def do_ab_eval(
    policy_path_a,
    policy_path_b,
    repo_id_a=None,
    repo_id_b=None,
    single_task="Grab the cube and drop it",
    num_episodes=20,
    episode_time_s=60,
    use_struggle_monitor=False,
    struggle_key_file=r"C:\Users\calle\Desktop\gem.txt",
    struggle_check_interval=2.0,
    struggle_threshold=0.6,
    struggle_model="gemini-pro",
    struggle_n_frames=12,
    struggle_score_mode="mean",
    struggle_warmup_s=10.0,
    struggle_temperatures=None,
    struggle_thresholds_a=None,
    struggle_thresholds_b=None,
    auto_switch=False,
    switch_duration=15.0,
    stats_csv=None,
):
    """
    A/B policy eval driven by manual q-key switches.

    Starts on policy A and stays on it until you press 'q', which switches
    to policy B. Press 'q' again to switch back to A. You control when
    switches happen — the script never alternates automatically.

    Both policies are loaded once at startup and stay in GPU memory for the
    whole run. The robot and camera also stay connected throughout, so there
    is no warmup delay between episodes.

    Args:
        policy_path_a:  HF model ID or local snapshot path for policy A.
                        Use resolve_policy_path() to get the local path if cached.
        policy_path_b:  Same for policy B.
        repo_id_a:      Dataset repo_id for policy A results. Auto-derived from
                        policy_path_a if not given.
        repo_id_b:      Same for policy B.
        single_task:             Task string written into both datasets.
        num_episodes:            Total episode cap across both policies combined.
        episode_time_s:          Max seconds per episode before it auto-ends.
        use_struggle_monitor:    If True, runs a LiveStruggleMonitor in the background.
                                 The monitor polls Gemini every struggle_check_interval
                                 seconds. When the EMA score exceeds struggle_threshold,
                                 the episode ends early. Episode stats go to stats_csv.
        struggle_key_file:       Path to a file containing the Gemini API key.
        struggle_check_interval: Seconds between Gemini assessments.
        struggle_threshold:      EMA score above which is_struggling() triggers.
        auto_switch:             If True (requires use_struggle_monitor=True), policy B
                                 takes over automatically when the monitor fires.
        switch_duration:         How many seconds policy B runs after an auto-switch
                                 (default 15 s). Arm holds position from where A got
                                 stuck. After B's stint the episode ends; both A's and
                                 B's frames are saved to their respective datasets.
                                 Next episode always starts on A from home position.
                                 Set to 0 to keep the old behaviour (episode ends
                                 immediately when monitor fires, no B intervention).
        stats_csv:               Optional path to a CSV for per-episode stats.
                                 Appended to (not overwritten) so partial runs survive.
    """
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
    from lerobot.datasets.utils import combine_feature_dicts
    from lerobot.datasets.video_utils import VideoEncodingManager
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.processor import make_default_processors
    from lerobot.processor.rename_processor import rename_stats
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.scripts.lerobot_record import record_loop
    from lerobot.utils.control_utils import (init_keyboard_listener,
                                              sanity_check_dataset_robot_compatibility)
    from lerobot.utils.utils import init_logging
    from lerobot.utils.visualization_utils import init_rerun

    # Derive dataset repo_ids from policy paths if not given explicitly.
    # e.g. "SkywalkerLi/smolvla-aug" -> "SkywalkerLi/eval_smolvla-aug"
    if repo_id_a is None:
        repo_id_a = repo_id_from_policy(policy_path_a)
    if repo_id_b is None:
        repo_id_b = repo_id_from_policy(policy_path_b)

    # Warn if CPU/RAM looks stressed before we open the camera
    _check_starvation()

    # Start rerun visualization on port 9090. Skip if already running so we
    # don't clear the view from a previous session.
    if _rerun_is_running():
        print("  Rerun viewer already running on :9090, skipping restart.")
    else:
        subprocess.run(["taskkill", "/f", "/im", "rerun.exe"], capture_output=True)
        subprocess.Popen(["rerun", "--serve-web"])
    # Wait for the gRPC port (9876) to be ready before connecting the SDK,
    # otherwise rr.spawn() starts a competing native viewer that steals the port.
    _wait_for_port(9876)
    threading.Thread(target=_wait_and_open_viewer, daemon=True).start()

    init_logging()
    init_rerun(session_name="recording", ip="127.0.0.1", port=9876)

    # ── ONE-TIME HARDWARE + PIPELINE SETUP ───────────────────────────────────
    # Everything here is created once and reused across all episodes for both
    # policies. Previously we called lerobot as a subprocess per episode which
    # reloaded weights (~20-30s) and re-warmed the camera (~8s) every time.

    # SO-101 arm on COM5 + OpenCV camera at index 1.
    # camera name "camera1" must match observation.images.camera1 in the policy.
    robot_cfg = SOFollowerRobotConfig(
        port="COM5",
        id="student_arm",
        use_degrees=True,
        cameras={
            "camera1": OpenCVCameraConfig(
                index_or_path=1, fps=30, width=640, height=480, warmup_s=2,
            )
        },
    )
    robot = SOFollower(robot_cfg)

    # Stateless pipelines shared by both policies
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Dataset feature schema derived from the robot hardware (joint names, image shape, etc.)
    # Both datasets use the same schema since they record from the same robot.
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    def _patch_vlm_cache():
        """Patch SmolVLMWithExpertModel so the 500M base VLM is loaded only once.

        Both policies use the same frozen SmolVLM2-500M backbone (train_expert_only=True
        means only the expert head was fine-tuned). Without this patch, the 1 GB of base
        weights are read from disk twice. With it, the second policy reuses the object
        already in VRAM — saving ~15-20s of load time.
        """
        try:
            from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
            _vlm_cache: dict = {}
            _orig_init = SmolVLMWithExpertModel.__init__

            def _cached_init(self, model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                              load_vlm_weights=True, **kwargs):
                if load_vlm_weights and model_id in _vlm_cache:
                    print(f"  [VLM cache] reusing {model_id} (skip reload)")
                    # init without weights (fast: random tensors only), then swap in cache
                    _orig_init(self, model_id=model_id, load_vlm_weights=False, **kwargs)
                    self.vlm = _vlm_cache[model_id]
                else:
                    _orig_init(self, model_id=model_id,
                               load_vlm_weights=load_vlm_weights, **kwargs)
                    if load_vlm_weights:
                        _vlm_cache[model_id] = self.vlm
                        print(f"  [VLM cache] cached {model_id} for reuse")

            SmolVLMWithExpertModel.__init__ = _cached_init
            print("  [VLM cache] patch applied — base VLM will load once only")
        except Exception as exc:
            print(f"  [VLM cache] patch skipped ({exc.__class__.__name__}: {exc})")

    _patch_vlm_cache()

    def _load_cfg(policy_path):
        """Load policy config and fix any cluster-specific paths baked into it."""
        _is_local = Path(policy_path).is_dir()
        cfg = PreTrainedConfig.from_pretrained(
            policy_path, local_files_only=_is_local
        )
        cfg.pretrained_path = policy_path
        cfg.device = "cuda"
        # Policies trained on a compute cluster may have an absolute path like
        # /scratch/gpfs/... baked into config.json for the VLM backbone.
        # Prefer the local HF cache snapshot; fall back to the hub ID only if
        # no cached snapshot is present.
        if hasattr(cfg, "vlm_model_name") and cfg.vlm_model_name.startswith("/"):
            _vlm_hf_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
            _vlm_cache = (
                Path.home()
                / ".cache/huggingface/hub"
                / "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
                / "snapshots"
            )
            _resolved = _vlm_hf_id  # default: resolve via hub
            if _vlm_cache.exists():
                _snaps = sorted(_vlm_cache.iterdir())
                if _snaps and any(_snaps[-1].glob("*.safetensors")):
                    _resolved = str(_snaps[-1])
            cfg.vlm_model_name = _resolved
        return cfg

    def _open_dataset(repo_id, policy_cfg):
        """
        Open or create a LeRobotDataset.
          - tasks.parquet exists       → valid dataset, resume (append episodes)
          - folder exists, no parquet  → incomplete/crashed run, delete and recreate
          - no folder                  → create fresh
        """
        dataset_path = Path.home() / ".cache/huggingface/lerobot" / repo_id
        resuming = (dataset_path / "meta" / "tasks.parquet").exists()
        if resuming:
            print(f"  [{repo_id}] Resuming existing dataset.")
            try:
                ds = LeRobotDataset(repo_id, root=dataset_path)
                ds.start_image_writer(num_processes=0, num_threads=4)
                sanity_check_dataset_robot_compatibility(ds, robot, 30, dataset_features)
            except Exception as e:
                print(f"  [{repo_id}] Corrupted dataset ({e.__class__.__name__}), wiping and recreating.")
                shutil.rmtree(dataset_path)
                resuming = False   # fall through to create branch
        if not resuming:
            if dataset_path.exists():
                print(f"  [{repo_id}] Incomplete folder found, starting fresh.")
                shutil.rmtree(dataset_path)
            ds = LeRobotDataset.create(
                repo_id,
                fps=30,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=4,  # background threads write video without blocking control loop
            )
        return ds

    # ── LOAD BOTH POLICIES UPFRONT ────────────────────────────────────────────
    # SmolVLA-500M takes ~20-30s to load. Doing it here means zero switching
    # overhead between episodes — both models sit in VRAM ready to go.

    print("\n--- Policy A ---")
    cfg_a = _load_cfg(policy_path_a)
    dataset_a = _open_dataset(repo_id_a, cfg_a)
    print("  Loading policy A weights...")
    policy_a = make_policy(cfg_a, ds_meta=dataset_a.meta)
    pre_a, post_a = make_pre_post_processors(
        policy_cfg=cfg_a,
        pretrained_path=policy_path_a,
        dataset_stats=rename_stats(dataset_a.meta.stats, {}),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    # Clear CUDA cache between the two model loads so the SVT encoder threads
    # from dataset_a creation don't race with policy_b CUDA initialisation.
    import torch, time as _time
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    _time.sleep(0.5)

    print("\n--- Policy B ---")
    cfg_b = _load_cfg(policy_path_b)
    dataset_b = _open_dataset(repo_id_b, cfg_b)
    print("  Loading policy B weights...")
    policy_b = make_policy(cfg_b, ds_meta=dataset_b.meta)
    pre_b, post_b = make_pre_post_processors(
        policy_cfg=cfg_b,
        pretrained_path=policy_path_b,
        dataset_stats=rename_stats(dataset_b.meta.stats, {}),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    # Connect robot + camera once. Keyboard listener starts here too.
    robot.connect()
    listener, events = init_keyboard_listener()       # lerobot's default keys (d, Escape, etc.)
    switch_listener = _start_switch_key_listener(events)  # our 'q' key on top
    events["switch_policy"] = False
    events["auto_switched"] = False

    counts = {"A": 0, "B": 0}  # episodes completed per policy
    current_label = "A"         # start on A; 'q' flips this each time

    # ── Struggle monitor (optional) ───────────────────────────────────────────
    # Load per-policy threshold objects once at startup.  Paths may be None
    # (falls back to pooled.json inside compute_struggle_score) or a str path.
    def _load_thresholds(path):
        if path is None:
            return None
        from struggle_monitor import Thresholds
        return Thresholds.load(path)

    _thresholds_a = _load_thresholds(struggle_thresholds_a)
    _thresholds_b = _load_thresholds(struggle_thresholds_b)

    monitor = None
    if use_struggle_monitor:
        from struggle_monitor import LiveStruggleMonitor
        monitor = LiveStruggleMonitor(
            key_file=struggle_key_file,
            model=struggle_model,
            check_interval=struggle_check_interval,
            interrupt_threshold=struggle_threshold,
            n_sample_frames=struggle_n_frames,
            score_mode=struggle_score_mode,
            warmup_s=struggle_warmup_s,
            temperatures=struggle_temperatures,
            thresholds=_thresholds_a,
        )
        monitor.start()
        # Wrap the robot so get_observation() feeds frames to the monitor
        # instead of opening a competing VideoCapture on the same camera.
        robot = _MonitorFeedingRobot(robot, monitor)
        print("  [StruggleMonitor] watching robot camera feed — will flag struggling episodes")

    # Shared mutable label — display thread reads this, main loop writes it.
    label_ref    = ["A"]
    display_stop = threading.Event()
    stats_queue  = None
    if monitor:
        import queue as _q
        stats_queue  = _q.Queue(maxsize=4)
        gui_proc     = threading.Thread(
            target=_stats_gui_process, args=(stats_queue, struggle_score_mode), daemon=True,
        )
        gui_proc.start()
        display_thread = threading.Thread(
            target=_live_display_loop,
            args=(monitor, label_ref, display_stop, struggle_threshold, stats_queue),
            daemon=True,
        )
        display_thread.start()

    def _struggle_watcher(stop_evt):
        """Background thread: triggers policy switch when monitor signals struggling.

        Only acts when auto_switch=True. When auto_switch=False the monitor runs
        passively — signals are visible in the GUI/terminal but episodes are not
        affected.
        """
        while not stop_evt.is_set():
            if (
                monitor
                and monitor.is_struggling()
                and monitor.episode_elapsed >= struggle_warmup_s
            ):
                if auto_switch:
                    print(f"\n  [StruggleMonitor] STRUGGLING — auto-switching policy after episode")
                    events["auto_switched"] = True
                    events["exit_early"] = True
                    break
                else:
                    print(f"\n  [StruggleMonitor] STRUGGLING (passive — press q to switch)", flush=True)
            stop_evt.wait(timeout=0.25)

    _pending_save: threading.Thread | None = None
    try:
        # Both VideoEncodingManagers stay open for the full run so their
        # background video-writing threads are always ready, regardless of
        # which dataset is currently active.
        with VideoEncodingManager(dataset_a):
            with VideoEncodingManager(dataset_b):
                for i in range(num_episodes):

                    _t_ep_start = _time.perf_counter()
                    _timings: dict[str, float] = {}
                    _b_ran = False   # set True if policy-B intervention fires

                    # ── PICK ACTIVE POLICY ────────────────────────────────────
                    # Every episode always starts on Policy A.
                    current_label = "A"
                    label_ref[0]  = "A"
                    held = events.get("_held_position", False)

                    label   = current_label
                    dataset = dataset_a if label == "A" else dataset_b
                    policy  = policy_a  if label == "A" else policy_b
                    pre     = pre_a     if label == "A" else pre_b
                    post    = post_a    if label == "A" else post_b
                    counts[label] += 1

                    # Reset events at the top of every episode so stray keypresses
                    # during go_home don't immediately exit the record_loop.
                    events["exit_early"] = False
                    events["rerecord_episode"] = False
                    events["switch_policy"] = False
                    events["auto_switched"] = False

                    if monitor:
                        monitor.resume_from_transfer()   # ensure never stuck paused
                        monitor.reset_signal()
                        if stats_queue is not None:
                            try:
                                stats_queue.put_nowait({"reset": True})
                            except Exception:
                                pass

                    # ── GO HOME (skipped after q) ──────────────────────────────
                    # Normally the arm returns home so every episode starts from
                    # the same known pose. If the previous episode ended with 'q',
                    # _held_position is True and we skip go_home — the arm and
                    # the object stay exactly where they are so the other policy
                    # gets a fair attempt from the identical starting state.
                    _t0 = _time.perf_counter()
                    if not held:
                        _go_home_with_robot(robot)
                    _timings["go_home"] = _time.perf_counter() - _t0

                    # Wait for the previous episode's saves to finish. Joining
                    # AFTER go_home lets saves run in parallel with the arm
                    # return, cutting the inter-episode dead time.
                    if _pending_save is not None:
                        _t0 = _time.perf_counter()
                        _pending_save.join()
                        _pending_save = None
                        _join_wait = _time.perf_counter() - _t0
                        if _join_wait > 0.5:
                            print(f"  [saves] waited {_join_wait:.1f}s for background saves to finish")

                    print(f"\n  Episode {i + 1}/{num_episodes} — "
                          f"Policy {label} (ep {counts[label]} for {label})")

                    # Per-episode watcher thread: ends episode early if struggling.
                    # Started AFTER go_home so stale frames in the rolling buffer
                    # (from the previous episode) don't trigger a false positive
                    # while the arm is returning home.
                    watcher_stop = threading.Event()
                    if monitor:
                        watcher = threading.Thread(
                            target=_struggle_watcher, args=(watcher_stop,), daemon=True
                        )
                        watcher.start()

                    # ── RUN EPISODE ───────────────────────────────────────────
                    # record_loop runs at 30 Hz: read obs -> policy forward pass
                    # -> send action -> store frame. Exits when control_time_s
                    # is reached or events["exit_early"] is set (d or q key).
                    _t0 = _time.perf_counter()
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=30,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        policy=policy,
                        preprocessor=pre,
                        postprocessor=post,
                        dataset=dataset,
                        control_time_s=episode_time_s,
                        single_task=single_task,
                        display_data=True,
                    )
                    _timings[f"record_loop_{label}"] = _time.perf_counter() - _t0

                    # Stop watcher thread and collect episode stats.
                    if monitor:
                        watcher_stop.set()
                        # ── VOTE-WAIT ───────────────────────────────────────────
                        # If S is above threshold but a Gemini vote is still
                        # in-flight (e.g. episode timer expired just as it fired),
                        # pause up to 2 × check_interval for the vote to land.
                        # This prevents missing a valid interrupt when A's time
                        # runs out a few seconds before Gemini responds.
                        if (
                            auto_switch
                            and not events.get("auto_switched")
                            and monitor.get_struggle_score() >= struggle_threshold
                            and monitor.is_vote_in_flight()
                        ):
                            _vote_deadline = _time.perf_counter() + struggle_check_interval * 2
                            print(
                                "  [VoteWait] S above threshold — waiting for in-flight vote...",
                                flush=True,
                            )
                            while monitor.is_vote_in_flight() and _time.perf_counter() < _vote_deadline:
                                _time.sleep(0.15)
                            if monitor.is_struggling():
                                events["auto_switched"] = True
                                print("  [VoteWait] Vote landed → INTERRUPT — switching to B.")
                            else:
                                print("  [VoteWait] Vote landed → no interrupt.")
                        ep_state = monitor.get_episode_state()
                        print(f"  [EpisodeState] {ep_state}")

                    # ── POLICY-B INTERVENTION ─────────────────────────────────
                    # When the monitor auto-triggered, run policy B for
                    # switch_duration seconds from exactly where A got stuck
                    # (arm holds position — no go_home). Frames go into B's
                    # dataset buffer as a separate episode. After B's stint the
                    # episode loop falls through to save both episodes and then
                    # the NEXT episode always resets to A from home.
                    if events.get("auto_switched") and auto_switch:
                        # How long B runs: fixed switch_duration, or the remainder
                        # of the episode budget — but always at least 20s so B
                        # gets a meaningful attempt even when the switch fires late.
                        if switch_duration > 0:
                            b_time_s = switch_duration
                        else:
                            elapsed_ep = _time.perf_counter() - _t_ep_start
                            b_time_s = max(20.0, episode_time_s - elapsed_ep)
                        print(f"\n  [AutoSwitch] Policy B intervening for {b_time_s:.0f}s "
                              f"from current position...")
                        label_ref[0] = "B"
                        if monitor:
                            monitor.set_thresholds(_thresholds_b)
                        events["exit_early"]    = False
                        events["auto_switched"] = False
                        if monitor:
                            # Snapshot the peak EMA and mark the switch on the
                            # chart BEFORE reset_signal() zeroes it out — ensures
                            # the chart shows the spike that caused the switch.
                            if stats_queue is not None:
                                try:
                                    _peak_intr = monitor.get_interrupt()
                                    stats_queue.put_nowait({
                                        "label":     label_ref[0],
                                        "ema":       monitor.get_struggle_score(),
                                        "threshold": struggle_threshold,
                                        "prob":      _peak_intr.get("interrupt_probability", 0.0),
                                        "votes":     _peak_intr.get("votes", []),
                                    })
                                    stats_queue.put_nowait({"switch_marker": True})
                                except Exception:
                                    pass
                            monitor.pause_for_transfer()
                            monitor.reset_signal(keep_timer=True)
                            monitor.suppress_judges(True)   # no votes needed after switch
                            monitor.resume_from_transfer()  # allow S+GUI updates during B's run
                        _t0 = _time.perf_counter()
                        record_loop(
                            robot=robot,
                            events=events,
                            fps=30,
                            teleop_action_processor=teleop_action_processor,
                            robot_action_processor=robot_action_processor,
                            robot_observation_processor=robot_observation_processor,
                            policy=policy_b,
                            preprocessor=pre_b,
                            postprocessor=post_b,
                            dataset=dataset_b,
                            control_time_s=b_time_s,
                            single_task=single_task,
                            display_data=True,
                        )
                        _timings["record_loop_B_intervention"] = _time.perf_counter() - _t0
                        # Keep label_ref as "B" until the next episode iteration
                        # sets it to "A" together with reset_signal + chart reset.
                        # Flipping early makes the display loop push cyan (A) with
                        # B's stale S value → flat ghost line after episode ends.
                        if monitor:
                            monitor.suppress_judges(False)  # re-enable for next A episode
                            monitor.set_thresholds(_thresholds_a)
                            monitor.resume_from_transfer()
                            b_ep_state = monitor.get_episode_state()
                            print(f"  [EpisodeState B] {b_ep_state}")
                        # B's frames will be saved in the background thread below.
                        b_frames = (
                            dataset_b.episode_buffer is not None
                            and dataset_b.episode_buffer.get("size", 0) > 0
                        )
                        _b_ran = True
                        # clear the auto_switched flag so the flip logic below
                        # doesn't also trigger
                        events["auto_switched"] = False

                    # ── SAVE EPISODES (background) ────────────────────────────
                    # Capture everything the thread needs by value so the next
                    # iteration can freely update local variables.
                    _sv_b_ran      = _b_ran
                    _sv_b_frames   = b_frames   if _b_ran else False
                    _sv_b_ep_state = b_ep_state if _b_ran else None
                    _sv_dataset    = dataset
                    _sv_label      = label
                    _sv_ep_state   = ep_state
                    _sv_switch_pol = events["switch_policy"]
                    _sv_ep_idx     = i + 1

                    def _bg_saves(
                        _b=_sv_b_ran, _bf=_sv_b_frames, _be=_sv_b_ep_state,
                        _ds=_sv_dataset, _lbl=_sv_label, _ep=_sv_ep_state,
                        _sw=_sv_switch_pol, _idx=_sv_ep_idx,
                    ):
                        # B's frames
                        if _b:
                            _time.sleep(2.5)
                            if _bf:
                                dataset_b.save_episode()
                                if stats_csv and monitor:
                                    _append_episode_stats_ab(
                                        stats_csv, dataset_b.num_episodes - 1, "B", _be)
                            else:
                                print("  WARNING: Policy B collected no frames — skipping save.")
                                if dataset_b.episode_buffer is not None:
                                    dataset_b.clear_episode_buffer()
                        # A's frames
                        _time.sleep(2.5)
                        a_ok = _ds.episode_buffer is not None and _ds.episode_buffer.get("size", 0) > 0
                        if a_ok:
                            if _sw:
                                # 'q' pressed mid-episode: copy to OTHER dataset too.
                                _other_lbl = "B" if _lbl == "A" else "A"
                                _other_ds  = dataset_b if _lbl == "A" else dataset_a
                                n_copied = _duplicate_buffer(_ds, _other_ds)
                                print(f"  [switch] Saving {n_copied} frames to both datasets.")
                                _ds.save_episode()
                                _other_ds.save_episode()
                                if stats_csv and monitor:
                                    _append_episode_stats_ab(
                                        stats_csv, _ds.num_episodes - 1, _lbl, _ep)
                                    _append_episode_stats_ab(
                                        stats_csv, _other_ds.num_episodes - 1, _other_lbl, _ep)
                            else:
                                _ds.save_episode()
                                if stats_csv and monitor:
                                    _append_episode_stats_ab(
                                        stats_csv, _ds.num_episodes - 1, _lbl, _ep)
                        else:
                            print(f"  WARNING: Episode {_idx} (Policy {_lbl}) collected no frames — skipping save.")
                            if _ds.episode_buffer is not None:
                                _ds.clear_episode_buffer()

                    _pending_save = threading.Thread(target=_bg_saves, daemon=True)
                    _pending_save.start()

                    _t_ep_total = _time.perf_counter() - _t_ep_start
                    _timing_str = "  ".join(f"{k}={v:.1f}s" for k, v in _timings.items())
                    print(f"  [TIMING ep {i+1}] total={_t_ep_total:.1f}s  |  {_timing_str}  (saves async)")

                    # ── FLIP POLICY IF q WAS PRESSED OR MONITOR TRIGGERED ─────
                    if events["switch_policy"]:
                        current_label = "B" if current_label == "A" else "A"
                        label_ref[0]  = current_label
                        if monitor:
                            monitor.set_thresholds(_thresholds_b if current_label == "B" else _thresholds_a)
                        print(f"  Now on Policy {current_label}.")
                    elif events.get("auto_switched"):
                        current_label = "B" if current_label == "A" else "A"
                        label_ref[0]  = current_label
                        print(f"  [AutoSwitch] Monitor triggered — now on Policy {current_label}.")
                    # Hold position on any policy switch (q or auto) — arm never goes
                    # home between policies so the scene stays identical for fair comparison.
                    events["_held_position"] = events["switch_policy"] or bool(events.get("auto_switched"))

                    if events["stop_recording"]:
                        break

    finally:
        display_stop.set()  # stop live cv2 display thread
        if monitor:
            monitor.stop()
        # Ensure any in-flight background save completes before finalize().
        if _pending_save is not None:
            _pending_save.join()
        # Park arm at home and clean up regardless of how the run ended.
        _go_home_with_robot(robot)
        robot.disconnect()
        listener.stop()
        switch_listener.stop()
        # finalize() writes index files and closes video writers.
        # Must be called on both so neither dataset is left incomplete.
        dataset_a.finalize()
        dataset_b.finalize()


def simulate_ab_on_dataset(
    dataset_id: str | None = None,
    existing_stats_csv: str | None = None,
    key_file: str = r"C:\Users\calle\Desktop\gem.txt",
    model: str = "gemini-2.5-flash",
    check_interval_s: float = 2.0,
    window_seconds: float = 20.0,
    interrupt_threshold: float = 0.5,
    ema_alpha: float = 0.25,
    results_csv: str | None = None,
    out_csv: str = "ab_simulated_stats.csv",
    video_dir: str | None = None,
    max_episodes: int | None = None,
):
    """Simulate automatic A/B policy switching on a LeRobot dataset.

    Two modes:
      1. existing_stats_csv — load a CSV already produced by test_on_dataset
         (e.g. train_stats.csv). No Gemini calls; just replays the interrupt
         decisions and adds a policy label. Use this for quick offline testing.
      2. dataset_id — runs test_on_dataset from scratch (makes Gemini API calls),
         then simulates the switching on top.

    Switch rule: starts on policy "A". Each time final_interrupt is True (monitor
    would have triggered), the NEXT episode flips to the other policy. This mirrors
    auto_switch=True in do_ab_eval on the live robot.

    Writes out_csv with all original columns plus "policy" and "auto_switched_after".
    Prints per-policy stats (success rate, clean pickup/drop, mean interrupt prob).

    Args:
        dataset_id:          HuggingFace dataset ID. Used when no existing_stats_csv.
        existing_stats_csv:  Path to a prior test_on_dataset CSV. Skips Gemini calls.
        key_file:            Gemini API key file (only used when running fresh).
        model:               Gemini model (only used when running fresh).
        check_interval_s:    Seconds between checks (only used when running fresh).
        window_seconds:      Rolling buffer length in seconds (fresh runs only).
        interrupt_threshold: EMA threshold for final_interrupt decision.
        ema_alpha:           EMA smoothing factor (fresh runs only).
        results_csv:         Ground-truth labels CSV (fresh runs only).
        out_csv:             Where to write the enriched A/B CSV.
        video_dir:           If set, render 3-panel videos per episode (fresh only).
        max_episodes:        Cap on episodes to process (fresh runs only).
    """
    import pandas as pd

    if existing_stats_csv is not None:
        print(f"Loading existing stats from {existing_stats_csv} ...")
        df = pd.read_csv(existing_stats_csv)
    elif dataset_id is not None:
        from struggle_monitor import test_on_dataset
        raw_csv = out_csv + ".raw.csv"
        df = test_on_dataset(
            dataset_id=dataset_id,
            key_file=key_file,
            model=model,
            check_interval_s=check_interval_s,
            window_seconds=window_seconds,
            interrupt_threshold=interrupt_threshold,
            ema_alpha=ema_alpha,
            results_csv=results_csv,
            out_csv=raw_csv,
            video_dir=video_dir,
            max_episodes=max_episodes,
        )
    else:
        raise ValueError("Provide either dataset_id or existing_stats_csv.")

    if df.empty:
        print("No episodes to process.")
        return df

    # ── Simulate A/B switching ────────────────────────────────────────────────
    # Replay per-episode interrupt decisions and assign a policy label.
    # When an interrupt would have fired, the NEXT episode flips to the other policy.
    #
    # Interrupt decision: peak_ema_score >= interrupt_threshold (if the column exists)
    # or final_interrupt from the CSV (pre-computed with whatever threshold was used
    # during the original run). Using peak_ema_score lets you experiment with
    # different thresholds without re-running Gemini.
    use_peak = "peak_ema_score" in df.columns
    policy_labels: list[str] = []
    switched_after: list[bool] = []
    cur = "A"
    for _, row in df.iterrows():
        policy_labels.append(cur)
        if use_peak:
            triggered = float(row.get("peak_ema_score", 0.0)) >= interrupt_threshold
        else:
            triggered = bool(row.get("final_interrupt", False))
        switched_after.append(triggered)
        if triggered:
            cur = "B" if cur == "A" else "A"

    df = df.copy()
    df.insert(1, "policy", policy_labels)
    df.insert(2, "auto_switched_after", switched_after)

    df.to_csv(out_csv, index=False)
    print(f"\nSaved A/B simulated results to {out_csv}  ({len(df)} episodes)")

    # ── Per-policy summary ────────────────────────────────────────────────────
    total_switches = sum(switched_after)
    print(f"\n=== Simulated A/B Summary (threshold={interrupt_threshold}) ===")
    print(f"  Total auto-switches : {total_switches}")
    for pol in ["A", "B"]:
        sub = df[df["policy"] == pol]
        n = len(sub)
        if n == 0:
            continue
        n_success    = int(sub["target_reached"].sum())      if "target_reached"      in sub else "?"
        n_clean_pick = int((sub["clean_pickup"] == True).sum()) if "clean_pickup"     in sub else "?"
        n_clean_drop = int((sub["clean_drop"]   == True).sum()) if "clean_drop"       in sub else "?"
        n_switched   = int(sub["auto_switched_after"].sum())
        mean_p       = sub["mean_interrupt_prob"].mean()     if "mean_interrupt_prob" in sub else float("nan")
        print(
            f"  Policy {pol} : {n:2d} episodes"
            f"  success={n_success}/{n}"
            f"  clean_pick={n_clean_pick}/{n}"
            f"  clean_drop={n_clean_drop}/{n}"
            f"  switched_away={n_switched}"
            f"  mean_p={mean_p:.3f}"
        )

    return df


# Cyan  (BGR) — student policy
_COLOR_STUDENT  = (255, 255,   0)
_COLOR_SUCCESS  = (  0, 255,   0)   # green — task accomplished
# Magenta (BGR) — teacher policy
_COLOR_TEACHER = (255,   0, 255)
_BORDER_PX     = 6
_STRIP_HEIGHT  = 130   # matches _render_episode_interrupt_video default


def simulate_ab_video(
    stats_csv: str,
    video_dir: str,
    out_video: str = "ab_simulated.mp4",
    interrupt_threshold: float = 0.5,
    show: bool = False,
    dataset_id: str | None = None,
) -> None:
    """Render a composite A/B simulation video from pre-computed interrupt stats.

    Reads stats_csv (output of test_on_dataset), replays the simulated A/B
    switching, and re-renders each episode's 3-panel interrupt video with:
      - A colored border around the camera section:
          cyan    = student policy  (Policy A)
          magenta = teacher policy  (Policy B)
      - A role badge in the top-right corner ("STUDENT" / "TEACHER").
      - A red "AUTO-SWITCH" banner from the exact frame the EMA threshold
        was crossed to the end of the episode (not just the last 2 s).

    All episodes are concatenated into a single output video.

    Args:
        stats_csv:           Path to a CSV that has episode_index, peak_ema_score,
                             and first_interrupt_ts columns (from test_on_dataset).
        video_dir:           Directory that contains episode_XXXX_interrupt.mp4 files.
        out_video:           Output .mp4 path.
        interrupt_threshold: peak_ema_score threshold for triggering a switch.
        show:                If True, display each frame in a real-time cv2 window
                             as well as writing to out_video. Press 'q' to quit early.
        dataset_id:          HuggingFace dataset ID used to load per-episode from_ts
                             (episode start time in the source video). Required to
                             compute the exact switch frame from first_interrupt_ts.
                             Uses local cache — no download. Falls back to end-of-
                             episode banner if not provided.
    """
    import cv2
    import glob as _glob
    import numpy as np
    import pandas as pd

    df = pd.read_csv(stats_csv)
    video_dir = Path(video_dir)
    out_path  = Path(out_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Build episode from_ts lookup (needed for exact switch frame) ──────────
    # first_interrupt_ts in the CSV is an absolute video timestamp.
    # The interrupt video for episode N starts at from_ts seconds into that video.
    # switch_frame = (first_interrupt_ts - from_ts) * fps
    ep_from_ts: dict[int, float] = {}
    if dataset_id is not None:
        try:
            from huggingface_hub import snapshot_download
            repo_dir = snapshot_download(repo_id=dataset_id, repo_type="dataset",
                                         local_files_only=True)
            meta_files = sorted(_glob.glob(f"{repo_dir}/meta/episodes/**/*.parquet",
                                           recursive=True))
            if meta_files:
                df_meta = pd.concat([pd.read_parquet(f) for f in meta_files],
                                    ignore_index=True)
                _cam_candidates = [
                    "videos/observation.images.camera1",
                    "videos/observation.images.front",
                    "videos/observation.images.top",
                    "videos/observation.images.camera_0",
                ]
                _cam = next(
                    (k for k in _cam_candidates
                     if f"{k}/from_timestamp" in df_meta.columns),
                    None,
                )
                if _cam:
                    for _, mrow in df_meta.iterrows():
                        ep_from_ts[int(mrow["episode_index"])] = float(
                            mrow[f"{_cam}/from_timestamp"]
                        )
                    print(f"  Loaded from_ts for {len(ep_from_ts)} episodes from meta.")
        except Exception as _e:
            print(f"  Warning: could not load episode from_ts ({_e}). "
                  f"Provide dataset_id for exact switch frames.")

    writer = None
    font   = cv2.FONT_HERSHEY_SIMPLEX

    n_total = 0
    vid_count         = 0   # sequential counter for rendered videos

    # Pre-count how many videos exist so we can show N/total in badge
    # Only render episodes where the threshold was actually crossed
    n_vids_expected = sum(
        1 for _, r in df.iterrows()
        if float(r.get("peak_ema_score", 0.0)) >= interrupt_threshold
        and (video_dir / f"episode_{int(r['episode_index']):04d}_interrupt.mp4").exists()
    )

    for _, row in df.iterrows():
        ep_idx    = int(row["episode_index"])
        peak_ema  = float(row.get("peak_ema_score", 0.0))
        switched  = peak_ema >= interrupt_threshold

        if not switched:
            continue  # skip episodes where teacher was never needed

        vid_path = video_dir / f"episode_{ep_idx:04d}_interrupt.mp4"
        if not vid_path.exists():
            print(f"  [sim] ep {ep_idx:04d}: video not found, skipping")
            continue

        vid_count += 1

        cap          = cv2.VideoCapture(str(vid_path))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_H        = H - 2 * _STRIP_HEIGHT
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ── Switch frame: exact moment EMA crossed threshold ──────────────────
        first_ts = row.get("first_interrupt_ts")
        from_ts  = ep_from_ts.get(ep_idx)
        if switched and first_ts is not None and not pd.isna(first_ts) and from_ts is not None:
            switch_frame = max(0, int((float(first_ts) - from_ts) * fps))
        else:
            switch_frame = total_frames  # never switch if not triggered

        if writer is None:
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
            )

        cam_top = _STRIP_HEIGHT
        cam_bot = _STRIP_HEIGHT + cam_H
        bt      = _BORDER_PX
        BANNER_FRAMES = int(fps * 2.5)   # show "TEACHER INTERVENES" for ~2.5 s

        n_total += 1

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # ── Two-phase model: STUDENT until threshold crossed, then TEACHER ─
            if switched and frame_idx >= switch_frame:
                border_color = _COLOR_TEACHER
                role_text    = "TEACHER"
            else:
                border_color = _COLOR_STUDENT
                role_text    = "STUDENT"

            # ── Colored border around camera section ──────────────────────────
            cv2.rectangle(frame,
                          (bt // 2,      cam_top + bt // 2),
                          (W - bt // 2,  cam_bot - bt // 2),
                          border_color, bt)

            # ── Role badge (top-right of camera section) ──────────────────────
            badge = f"[{vid_count}/{n_vids_expected}] ep {ep_idx}  {role_text}"
            (tw, th), _ = cv2.getTextSize(badge, font, 0.55, 2)
            pad = 6
            bx0 = W - tw - 2 * pad - bt - 4
            by0 = cam_top + bt + 4
            bx1 = W - bt - 4
            by1 = by0 + th + 2 * pad
            overlay = frame.copy()
            cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.putText(frame, badge, (bx0 + pad, by1 - pad),
                        font, 0.55, border_color, 2, cv2.LINE_AA)

            # ── "TEACHER INTERVENES" banner for first ~2.5 s after switch ─────
            if (switched
                    and switch_frame <= frame_idx < switch_frame + BANNER_FRAMES):
                banner = "TEACHER INTERVENES"
                (bw, bh), _ = cv2.getTextSize(banner, font, 1.1, 3)
                cam_mid = cam_top + cam_H // 2
                tx = (W - bw) // 2
                ty = cam_mid + bh // 2
                ov3 = frame.copy()
                cv2.rectangle(ov3,
                              (tx - 16, ty - bh - 14),
                              (tx + bw + 16, ty + 14),
                              (80, 0, 80), -1)
                cv2.addWeighted(ov3, 0.82, frame, 0.18, 0, frame)
                cv2.putText(frame, banner, (tx, ty),
                            font, 1.1, _COLOR_TEACHER, 3, cv2.LINE_AA)

            if show:
                cv2.imshow("A/B Eval Preview", frame)
                if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
                    show = False
                    cv2.destroyAllWindows()

            writer.write(frame)
            frame_idx += 1

        cap.release()
        print(f"  [{vid_count}/{n_vids_expected}] ep {ep_idx:04d}  peak_ema={peak_ema:.3f}"
              f"  switch={'@' + str(switch_frame) + 'f' if switched else 'no'}")

    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if writer:
        print(f"\nSaved: {out_path}  ({n_total} episodes with teacher intervention)")
        print(f"  (interrupt_threshold={interrupt_threshold})")
    else:
        print("No episodes rendered — check that video_dir contains episode_XXXX_interrupt.mp4 files.")


# Entry point is better_code.py — run via:
#   python better_code.py            # live A/B eval on robot
#   python better_code.py --simulate # offline simulation video
