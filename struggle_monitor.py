"""
Live policy struggle monitor using Gemini.

A background thread watches a rolling window of camera frames and
periodically asks Gemini whether the robot is struggling. Your control
loop just calls is_struggling() — no Gemini latency on the hot path.

Quick start
───────────
    monitor = LiveStruggleMonitor(key_file=r"C:/Users/calle/Desktop/gem.txt")
    monitor.start()

    # inside your control loop (called every step):
    monitor.push_frame(bgr_frame)        # numpy BGR uint8 from camera
    if monitor.is_struggling():
        print(monitor.get_signal())      # {"struggling", "confidence", "reason"}
        switch_policy()

    monitor.stop()
"""

import glob
import json
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from huggingface_hub import snapshot_download

from analyze_rollout import load_api_key

# ── Prompts ───────────────────────────────────────────────────────────────────

STRUGGLE_PROMPT = """You are watching a sequence of frames from a live robot arm
doing a pick-and-place task. The frames are evenly sampled from the last few seconds.
{stability_block}
Is the robot currently struggling?

Struggling signals:
  - Gripper missed the block or has an unstable / tilted grip
  - Block is rotating or slipping inside the gripper
  - Arm making repeated small corrections without making progress
  - Arm oscillating, stuck, or moving erratically
  - Block has been dropped or is about to fall
  - Lyapunov V trend is DIVERGING (state moving away from goal) — strong signal
  - Jerk is elevated far above baseline — arm is erratic

Not struggling:
  - Arm moving smoothly and purposefully toward a clear goal
  - Block held firmly and stably
  - Clean, confident approach to pick or drop point
  - Lyapunov V trend is CONVERGING — state closing in on goal

Return ONLY JSON with fields: struggling (bool), confidence (0-1), reason (one sentence).
"""

INTERRUPT_PROMPT = """You are a supervisor watching 20 seconds of a robot arm doing a
pick-and-place task. The frames span the FULL 20-second window so you can see the
outcome of any attempt — use this to count successes and failures.

IMPORTANT CONTEXT: The robot arm ALWAYS starts each episode with an EMPTY, OPEN gripper
in a neutral rest position above the workspace. The very first frames will show the arm
approaching the block with an open gripper — this is normal and NOT a drop event.
Only count gripper opens as drop_attempts AFTER the gripper has first successfully closed
on the block.

{stability_block}
{gripper_block}

Your job is to rate the PROBABILITY that a human operator should interrupt and switch
to a different policy RIGHT NOW (at the END of this 20-second window).

Count carefully from the frames:
  pickup_attempts : how many times did the gripper close on or near the block?
                    (1 = one clean grasp, 2+ = retries or failed attempts)
  drop_attempts   : how many times did the gripper open to release the block AFTER
                    a successful pickup? (1 = one clean drop, 2+ = retries)
                    Do NOT count the initial open-gripper approach as a drop attempt.

NORMAL BEHAVIOUR — do NOT penalise these:
  - A single re-grasp: the gripper briefly re-closes to improve grip (pickup_attempts = 2 is fine)
  - A single drop correction or push motion to seat the block (drop_attempts = 2 is fine)
  These are expected and represent good adaptive behaviour, not failure.

interrupt_probability scoring guide:
  0.0 – 0.2  Clean execution. 1–2 pickups/drops, block held stably, converging trend.
  0.2 – 0.4  Minor hiccup (one slip, one jerk spike) but recovering — low urgency.
  0.4 – 0.6  Messy but possibly recovering. 3 attempts or mildly diverging trend.
  0.6 – 0.8  Clearly struggling. 4+ attempts, sustained divergence, or block dropped.
  0.8 – 1.0  Policy has failed. No progress, block lost, looping, or totally erratic.

Return ONLY JSON with fields:
  interrupt_probability (float 0-1),
  pickup_attempts (int),
  drop_attempts (int),
  reason (one sentence explaining the score).
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "struggling": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason":     {"type": "string"},
    },
    "required": ["struggling", "confidence", "reason"],
}

INTERRUPT_SCHEMA = {
    "type": "object",
    "properties": {
        "interrupt_probability": {"type": "number"},
        "pickup_attempts":       {"type": "integer"},
        "drop_attempts":         {"type": "integer"},
        "reason":                {"type": "string"},
    },
    "required": ["interrupt_probability", "pickup_attempts", "drop_attempts", "reason"],
}

# ── Default signals ───────────────────────────────────────────────────────────

_DEFAULT_SIGNAL = {"struggling": False, "confidence": 0.0, "reason": "no assessment yet"}
_DEFAULT_INTERRUPT = {
    "interrupt_probability": 0.0,
    "pickup_attempts": 0,
    "drop_attempts": 0,
    "reason": "no assessment yet",
}

# Time ramp: scale interrupt probability linearly from 0 → 1 over the first N seconds.
# Prevents early-episode checks from triggering high probabilities before the arm
# has had a chance to attempt the task.
TIME_RAMP_END_S: float = 30.0

# Probability cap: Gemini's raw output is capped at P_CAP_MIN early in the episode.
# The cap rises linearly to P_CAP_MAX between TIME_RAMP_END_S and CAP_RAMP_END_S,
# allowing very high probabilities only after sustained struggle.
P_CAP_MIN:      float = 0.75   # max allowed p before CAP_RAMP_END_S
P_CAP_MAX:      float = 0.90   # max allowed p after CAP_RAMP_END_S
CAP_RAMP_END_S: float = 60.0   # seconds at which cap reaches P_CAP_MAX


# ── Core assessment function ──────────────────────────────────────────────────

def _build_gripper_block(gripper: np.ndarray, threshold: float = 20.0) -> str:
    """Format the raw gripper position signal as a text block for the prompt.

    Also numerically counts pickup/drop attempts so Gemini can cross-check.
    """
    n = len(gripper)
    if n < 2:
        return ""

    ups   = [i for i in range(1, n) if gripper[i-1] < threshold <= gripper[i]]
    downs = [i for i in range(1, n) if gripper[i-1] >= threshold > gripper[i]]

    # Summarise gripper trajectory as compact string (sampled at ~20 points)
    step = max(1, n // 20)
    sampled = gripper[::step]
    trace = "  ".join(f"{v:.1f}" for v in sampled)

    return (
        f"\n--- GRIPPER SIGNAL (last {n} frames, threshold={threshold}°) ---\n"
        f"  Closes (pickup events) detected: {len(downs)}\n"
        f"  Opens  (drop events)   detected: {len(ups)}\n"
        f"  Sampled trace (deg): {trace}\n"
        f"--- END GRIPPER ---\n"
    )


def _build_stability_block(actions: np.ndarray, states: np.ndarray) -> str:
    """Compute rolling Lyapunov/jerk metrics and format as a text block for the prompt.

    Args:
        actions: (T, D_a) float32 array of recent actions.
        states:  (T, D_s) float32 array of recent observation states.

    Returns:
        Formatted stability context string to inject into STRUGGLE_PROMPT.
    """
    if len(actions) < 5:
        return ""

    # Jerk
    delta     = np.diff(actions, axis=0)
    jerk      = np.linalg.norm(delta, axis=1)
    mean_jerk = float(jerk.mean())
    cur_jerk  = float(jerk[-1])
    jerk_ratio = cur_jerk / (mean_jerk + 1e-9)
    jerk_tag  = f"HIGH ({jerk_ratio:.1f}x baseline)" if jerk_ratio > 2.0 else "normal"

    # Lyapunov  V(t) = ||s_t - s_latest||^2  (goal approx = most recent state)
    s_ref = states[-1]
    V     = np.sum((states - s_ref) ** 2, axis=1)
    dV    = np.diff(V)
    viol_rate   = float((dV > 0).mean())
    lyap_trend  = "DIVERGING" if viol_rate > 0.5 else "CONVERGING"
    lyap_detail = f"V increased {viol_rate*100:.0f}% of recent steps"

    # Discontinuities
    thresh = mean_jerk + 3.0 * float(jerk.std())
    n_disc = int((jerk > thresh).sum())

    return (
        f"\n--- STABILITY METRICS (last {len(actions)} frames) ---\n"
        f"  Jerk          : mean={mean_jerk:.2f}  current={cur_jerk:.2f}  [{jerk_tag}]\n"
        f"  Lyapunov trend: {lyap_trend}  ({lyap_detail})\n"
        f"  Discontinuities in window: {n_disc}\n"
        f"--- END METRICS ---\n"
    )


def assess_frames(
    frames: list[np.ndarray],
    client: genai.Client,
    model: str = "gemini-2.5-flash",
    jpeg_quality: int = 75,
    actions: np.ndarray | None = None,
    states:  np.ndarray | None = None,
) -> dict:
    """Send frames (+ optional stability metrics) to Gemini for a struggle assessment.

    Args:
        frames:       List of BGR uint8 numpy arrays (camera frames).
        client:       Authenticated genai.Client.
        model:        Gemini model ID.
        jpeg_quality: JPEG compression quality (lower = faster upload).
        actions:      (T, D_a) recent action array for stability context.
        states:       (T, D_s) recent state array for stability context.

    Returns:
        {"struggling": bool, "confidence": float, "reason": str}
    """
    if not frames:
        return _DEFAULT_SIGNAL.copy()

    # Build stability block if data is available
    stability_block = ""
    if actions is not None and states is not None and len(actions) >= 5:
        stability_block = _build_stability_block(actions, states)

    prompt = STRUGGLE_PROMPT.format(stability_block=stability_block)

    # Encode frames as inline JPEG image parts
    parts = []
    for frame in frames:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            continue
        parts.append(types.Part.from_bytes(data=bytes(buf), mime_type="image/jpeg"))

    if not parts:
        return _DEFAULT_SIGNAL.copy()

    parts.append(types.Part.from_text(text=prompt))

    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.2,
        ),
    )

    return json.loads(response.text)


def assess_interrupt(
    frames: list[np.ndarray],
    client: genai.Client,
    model: str = "gemini-2.5-flash",
    jpeg_quality: int = 75,
    actions: np.ndarray | None = None,
    states:  np.ndarray | None = None,
    gripper: np.ndarray | None = None,
    gripper_threshold: float = 20.0,
) -> dict:
    """Send 20 seconds of frames + motor/convergence/gripper data to Gemini.

    Returns interrupt_probability (0-1), pickup_attempts, drop_attempts, reason.
    """
    if not frames:
        return _DEFAULT_INTERRUPT.copy()

    stability_block = ""
    if actions is not None and states is not None and len(actions) >= 5:
        stability_block = _build_stability_block(actions, states)

    gripper_block = ""
    if gripper is not None and len(gripper) >= 2:
        gripper_block = _build_gripper_block(gripper, gripper_threshold)

    prompt = INTERRUPT_PROMPT.format(
        stability_block=stability_block,
        gripper_block=gripper_block,
    )

    parts = []
    for frame in frames:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            continue
        parts.append(types.Part.from_bytes(data=bytes(buf), mime_type="image/jpeg"))

    if not parts:
        return _DEFAULT_INTERRUPT.copy()

    parts.append(types.Part.from_text(text=prompt))

    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=INTERRUPT_SCHEMA,
            temperature=0.2,
        ),
    )

    return json.loads(response.text)


# ── Background monitor ────────────────────────────────────────────────────────

class LiveStruggleMonitor:
    """Continuously watches a 20-second rolling buffer and polls Gemini.

    Uses the richer INTERRUPT_PROMPT: Gemini counts pickup/drop attempts,
    reads convergence data, and outputs interrupt_probability (0-1).
    An EMA struggle_score smooths out transient spikes.
    is_struggling() returns True when the EMA score exceeds the threshold.

    Args:
        key_file:            Path to Gemini API key file (or None to use env var).
        model:               Gemini model to use.
        check_interval:      How often (seconds) to call Gemini.
        buffer_seconds:      Rolling buffer length — default 20s for full context.
        n_sample_frames:     Frames subsampled per call.
        fps:                 Camera frame rate (for buffer sizing).
        interrupt_threshold: EMA score above which is_struggling() → True.
        ema_alpha:           EMA decay for struggle_score (higher = more reactive).
        gripper_col_idx:     Index into observation.state for gripper position.
        gripper_threshold:   Degrees above which gripper is considered closed.
    """

    def __init__(
        self,
        key_file: str | None = None,
        model: str = "gemini-2.5-flash",
        check_interval: float = 2.0,
        buffer_seconds: float = 20.0,
        n_sample_frames: int = 12,
        fps: float = 30.0,
        interrupt_threshold: float = 0.5,
        ema_alpha: float = 0.4,
        gripper_col_idx: int = 5,
        gripper_threshold: float = 20.0,
    ):
        self._client             = genai.Client(api_key=load_api_key(key_file))
        self._model              = model
        self._check_interval     = check_interval
        self._n_sample           = n_sample_frames
        self._threshold          = interrupt_threshold
        self._ema_alpha          = ema_alpha
        self._gripper_col        = gripper_col_idx
        self._gripper_thresh     = gripper_threshold

        buffer_size              = int(buffer_seconds * fps)
        self._buffer: deque      = deque(maxlen=buffer_size)
        self._action_buf: deque  = deque(maxlen=buffer_size)
        self._state_buf:  deque  = deque(maxlen=buffer_size)
        self._lock               = threading.Lock()

        self._signal: dict       = _DEFAULT_SIGNAL.copy()
        self._interrupt: dict    = _DEFAULT_INTERRUPT.copy()
        self._struggle_score: float = 0.0
        self._stop_event         = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture_thread: threading.Thread | None = None
        self._capture_fps: float = fps

    # ── Public API ────────────────────────────────────────────────────────────

    def push_frame(self, frame: np.ndarray) -> None:
        """Add a BGR camera frame to the rolling buffer. Call every control step."""
        with self._lock:
            self._buffer.append(frame.copy())

    def push_observation(self, action: np.ndarray, state: np.ndarray) -> None:
        """Feed the latest action and observation state for stability metrics.

        Call alongside push_frame() every control step.

        Args:
            action: 1-D float32 array of joint actions.
            state:  1-D float32 array of observation state.
        """
        with self._lock:
            self._action_buf.append(np.asarray(action, dtype=np.float32))
            self._state_buf.append(np.asarray(state, dtype=np.float32))

    def is_struggling(self) -> bool:
        """True when the EMA struggle_score exceeds the interrupt threshold."""
        return self._struggle_score >= self._threshold

    def get_signal(self) -> dict:
        """Return the full latest signal: {struggling, confidence, reason}."""
        return self._signal.copy()

    def get_interrupt(self) -> dict:
        """Return latest interrupt assessment: {interrupt_probability, pickup_attempts, drop_attempts, reason}."""
        return self._interrupt.copy()

    def get_struggle_score(self) -> float:
        """Return current EMA struggle score (0-1)."""
        return self._struggle_score

    def reset_signal(self) -> None:
        """Clear signals and EMA at the start of each new episode."""
        self._signal         = _DEFAULT_SIGNAL.copy()
        self._interrupt      = _DEFAULT_INTERRUPT.copy()
        self._struggle_score = 0.0

    def start(self) -> None:
        """Start the background Gemini monitoring thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[StruggleMonitor] started  model={self._model}"
              f"  interval={self._check_interval}s"
              f"  threshold={self._threshold}")

    def stop(self) -> None:
        """Stop the background thread (and capture thread if running)."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5)
        print("[StruggleMonitor] stopped")

    def start_capture(self, camera_index: int = 1) -> None:
        """Open a dedicated OpenCV capture and feed frames into the buffer automatically.

        Use this when the robot's camera is shared and you can't push frames
        manually, or when integrating alongside a blocking control loop.
        The capture runs in its own daemon thread.

        Args:
            camera_index: cv2.VideoCapture index (same as robot camera, e.g. 1).
        """
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(camera_index,),
            daemon=True,
        )
        self._capture_thread.start()
        print(f"[StruggleMonitor] capture thread started  camera_index={camera_index}")

    def stop_capture(self) -> None:
        """Signal the capture thread to stop (also called by stop())."""
        self._stop_event.set()

    # ── Background loops ──────────────────────────────────────────────────────

    def _capture_loop(self, camera_index: int) -> None:
        """Continuously grab frames from the camera and push to buffer."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[StruggleMonitor] WARNING: could not open camera {camera_index}")
            return
        delay = 1.0 / self._capture_fps
        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if ok:
                self.push_frame(frame)
            self._stop_event.wait(timeout=delay)
        cap.release()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            start = time.time()

            with self._lock:
                buf     = list(self._buffer)
                actions = (
                    np.stack(list(self._action_buf))
                    if len(self._action_buf) >= 5 else None
                )
                states = (
                    np.stack(list(self._state_buf))
                    if len(self._state_buf) >= 5 else None
                )
                gripper = (
                    states[:, self._gripper_col]
                    if states is not None else None
                )

            if len(buf) >= self._n_sample:
                frames = self._subsample(buf, self._n_sample)
                try:
                    result = assess_interrupt(
                        frames, self._client, self._model,
                        actions=actions, states=states,
                        gripper=gripper,
                        gripper_threshold=self._gripper_thresh,
                    )
                    self._interrupt = result
                    # EMA update
                    p = float(result["interrupt_probability"])
                    self._struggle_score = (
                        self._ema_alpha * p
                        + (1 - self._ema_alpha) * self._struggle_score
                    )
                    status = "INTERRUPT" if self.is_struggling() else "ok     "
                    print(
                        f"[StruggleMonitor] {status}"
                        f"  p={p:.2f}  ema={self._struggle_score:.2f}"
                        f"  picks={result['pickup_attempts']}"
                        f"  drops={result['drop_attempts']}"
                        f"  | {result['reason']}"
                    )
                except Exception as e:
                    print(f"[StruggleMonitor] Gemini error: {e}")

            elapsed = time.time() - start
            self._stop_event.wait(timeout=max(0.0, self._check_interval - elapsed))

    @staticmethod
    def _subsample(frames: list, n: int) -> list:
        """Evenly subsample n frames from a list."""
        if len(frames) <= n:
            return frames
        indices = [int(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
        return [frames[i] for i in indices]


# ── Interrupt probability video renderer ──────────────────────────────────────

def _draw_interrupt_hud(
    frame: np.ndarray,
    interrupt_prob: float,
    ema_score: float,
    pickup_attempts: int,
    drop_attempts: int,
    reason: str,
    threshold: float = 0.5,
    time_factor: float = 1.0,
    p_cap: float = 0.90,
) -> np.ndarray:
    """Draw a probability gauge at the top-left of a BGR frame (in-place copy)."""
    frame = frame.copy()
    font  = cv2.FONT_HERSHEY_SIMPLEX

    # Panel dimensions — taller to fit 3 reason lines
    pw, ph = 260, 130
    x0, y0 = 8, 8

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Colour: green → yellow → red based on probability
    r = int(min(255, interrupt_prob * 2 * 255))
    g = int(min(255, (1 - interrupt_prob) * 2 * 255))
    prob_color = (0, g, r)
    ema_color  = (0, 200, 80) if ema_score < threshold else (0, 60, 220)

    # Probability bar at bottom of panel
    bar_x0  = x0 + 4
    bar_y0  = y0 + ph - 14
    bar_len = int((pw - 8) * interrupt_prob)
    cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + pw - 8, bar_y0 + 8), (50, 50, 50), -1)
    if bar_len > 0:
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + bar_len, bar_y0 + 8), prob_color, -1)
    thr_x = bar_x0 + int((pw - 8) * threshold)
    cv2.line(frame, (thr_x, bar_y0 - 2), (thr_x, bar_y0 + 10), (255, 255, 255), 1)

    # Row 1: scaled prob + EMA on same line
    cv2.putText(frame, f"p={interrupt_prob:.2f}",
                (x0 + 6, y0 + 20), font, 0.52, prob_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"ema={ema_score:.2f}",
                (x0 + 90, y0 + 20), font, 0.45, ema_color, 1, cv2.LINE_AA)
    # Time ramp / cap indicator — shown until fully ramped and cap at max
    if time_factor < 1.0 or p_cap < P_CAP_MAX:
        label = f"×{time_factor:.0%} ≤{p_cap:.2f}"
        cv2.putText(frame, label, (x0 + 155, y0 + 20), font, 0.36, (120, 120, 120), 1, cv2.LINE_AA)
    # Row 2: cumulative attempt counts
    cv2.putText(frame, f"picks {pickup_attempts}   drops {drop_attempts}",
                (x0 + 6, y0 + 38), font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    # Rows 3-5: Gemini reason text — wrap at ~38 chars, up to 3 lines
    reason_short = (reason[:110] + "..") if len(reason) > 112 else reason
    words, line, lines = reason_short.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > 38:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)
    for i, ln in enumerate(lines[:3]):
        cv2.putText(frame, ln, (x0 + 6, y0 + 56 + i * 16),
                    font, 0.34, (210, 210, 210), 1, cv2.LINE_AA)

    return frame


def _render_episode_interrupt_video(
    ep_idx: int,
    checks: list[dict],
    actions: np.ndarray,
    states: np.ndarray,
    gripper: np.ndarray,
    vid_path: Path,
    from_ts: float,
    to_ts: float,
    out_path: Path,
    interrupt_threshold: float = 0.5,
    gripper_threshold: float = 20.0,
    strip_height: int = 130,
) -> None:
    """Render one 3-panel interrupt video using pre-loaded episode data.

    No dataset re-loading — caller passes actions/states/gripper already in memory.
    """
    from stability_eval import _render_stability_strip
    from batch_detect_phases import _render_gripper_strip, detect_phases

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    delta       = np.diff(actions, axis=0)
    jerk        = np.linalg.norm(delta, axis=1)
    s_final     = states[-1]
    V           = np.sum((states - s_final) ** 2, axis=1)
    dV          = np.diff(V)
    mean_jerk   = jerk.mean()
    disc_frames = list(np.where(jerk > mean_jerk + 3 * jerk.std())[0].astype(int))

    pickup_frame, drop_frame = detect_phases(gripper, gripper_threshold)

    # Open original video and seek directly — avoids ffmpeg clip-extraction edge cases
    print(f"  rendering ep {ep_idx:4d} ...")
    cap = cv2.VideoCapture(str(vid_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, from_ts * 1000)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    top_strip = _render_stability_strip(jerk, dV, disc_frames, W, strip_height)
    bot_strip = _render_gripper_strip(gripper, pickup_frame, drop_frame,
                                      gripper_threshold, W, strip_height)

    total_H = strip_height + H + strip_height
    writer  = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (W, total_H))

    # Make picks/drops monotonically increasing across the episode
    max_picks = max_drops = 0
    mono_checks = []
    for ck in sorted(checks, key=lambda c: c["frame_idx"]):
        max_picks = max(max_picks, ck["picks"])
        max_drops = max(max_drops, ck["drops"])
        mono_checks.append({**ck, "picks": max_picks, "drops": max_drops})

    check_by_frame      = {ck["frame_idx"]: ck for ck in mono_checks}
    sorted_check_frames = sorted(check_by_frame.keys())

    # Mark the highest-probability frame permanently on the stability strip
    T_jerk    = len(jerk)
    T_gripper = len(gripper)
    if mono_checks:
        peak_ck  = max(mono_checks, key=lambda c: c["p"])
        peak_cx  = int(np.clip(peak_ck["frame_idx"] / max(T_jerk, 1) * W, 0, W - 1))
        cv2.line(top_strip, (peak_cx, 0), (peak_cx, strip_height), (0, 100, 255), 2)
        cv2.putText(top_strip, f"peak p={peak_ck['p']:.2f}",
                    (min(peak_cx + 3, W - 95), strip_height - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 255), 1, cv2.LINE_AA)

    frame_idx         = 0
    cur_check         = {"p": 0.0, "ema": 0.0, "picks": 0, "drops": 0, "reason": ""}
    cur_check_key     = -1          # frame_idx of the active check
    check_start_frame = 0
    type_out_frames   = int(fps * 1.5)  # fully reveal text within 1.5 s of each check
    font              = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_ms / 1000.0 > to_ts + 0.1:
            break

        # Advance to the most recent check at or before this frame
        new_key = -1
        for cf in sorted_check_frames:
            if cf <= frame_idx:
                new_key = cf
        if new_key != cur_check_key:
            cur_check         = check_by_frame[new_key] if new_key >= 0 else cur_check
            cur_check_key     = new_key
            check_start_frame = frame_idx

        # Typewriter effect: reveal reason text progressively after each new check
        reason_full  = cur_check["reason"]
        elapsed      = frame_idx - check_start_frame
        chars        = max(1, int(elapsed / max(type_out_frames, 1) * len(reason_full) + 0.5))
        live_reason  = reason_full[:min(chars, len(reason_full))]

        frame = _draw_interrupt_hud(
            frame,
            interrupt_prob=cur_check["p"],
            ema_score=cur_check["ema"],
            pickup_attempts=cur_check["picks"],
            drop_attempts=cur_check["drops"],
            reason=live_reason,
            threshold=interrupt_threshold,
            time_factor=cur_check.get("time_factor", 1.0),
            p_cap=cur_check.get("p_cap", P_CAP_MAX),
        )
        cv2.putText(frame, f"Episode {ep_idx}", (W - 130, 22),
                    font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        top = top_strip.copy()
        cx  = int(np.clip(frame_idx / max(T_jerk, 1) * W, 0, W - 1))
        cv2.line(top, (cx, 0), (cx, strip_height), (0, 255, 255), 2)
        if frame_idx < T_jerk:
            cv2.putText(top, f"jerk {jerk[frame_idx]:.1f}",
                        (min(cx + 4, W - 80), 16), font, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        bot = bot_strip.copy()
        cx2 = int(np.clip(frame_idx / max(T_gripper - 1, 1) * W, 0, W - 1))
        cv2.line(bot, (cx2, 0), (cx2, strip_height), (0, 255, 255), 2)
        if frame_idx < T_gripper:
            cv2.putText(bot, f"{gripper[frame_idx]:.1f}",
                        (min(cx2 + 4, W - 70), 16), font, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        writer.write(np.vstack([top, frame, bot]))
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"           -> {out_path}")


# ── Batch dataset test ────────────────────────────────────────────────────────

def test_on_dataset(
    dataset_id: str = "nc8304/eval_smolvla-phase-split_combined",
    key_file: str | None = None,
    model: str = "gemini-2.5-flash",
    check_interval_s: float = 2.0,
    window_seconds: float = 20.0,
    n_sample_frames: int = 12,
    fps: float = 30.0,
    interrupt_threshold: float = 0.5,
    ema_alpha: float = 0.4,
    gripper_col_idx: int = 5,
    gripper_threshold: float = 20.0,
    results_csv: str | None = None,
    out_csv: str = "struggle_test_results.csv",
    video_dir: str | None = None,
    max_episodes: int | None = None,
) -> pd.DataFrame:
    """Run the new interrupt monitor over every episode using 20-second windows.

    Gemini receives: 12 subsampled frames from a 20s rolling window,
    stability metrics (jerk + Lyapunov), and the gripper signal (for attempt
    counting). It returns interrupt_probability (0-1), pickup_attempts,
    drop_attempts, and reason.

    An EMA struggle_score is maintained per episode. Results are compared to
    known labels from results_csv.

    Args:
        dataset_id:          HuggingFace dataset ID.
        key_file:            Path to Gemini API key.
        model:               Gemini model to use.
        check_interval_s:    Seconds between Gemini calls (video time).
        window_seconds:      Rolling buffer length — 20s gives Gemini full attempt context.
        n_sample_frames:     Frames subsampled per call.
        fps:                 Assumed camera frame rate.
        interrupt_threshold: EMA score above which episode is flagged.
        ema_alpha:           EMA smoothing factor.
        gripper_col_idx:     Index into observation.state for gripper position.
        gripper_threshold:   Degrees above which gripper is closed.
        results_csv:         Path to labeled results.csv.
        out_csv:             Where to save per-episode results.
        video_dir:           If set, render 3-panel interrupt videos here.
    """
    print(f"Downloading dataset {dataset_id} ...")
    repo_dir   = snapshot_download(repo_id=dataset_id, repo_type="dataset")
    data_files = sorted(glob.glob(f"{repo_dir}/data/**/*.parquet", recursive=True))
    meta_files = sorted(glob.glob(f"{repo_dir}/meta/episodes/**/*.parquet", recursive=True))
    df_data    = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    df_meta    = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)

    # Labels
    if results_csv is None:
        script_dir = Path(__file__).parent
        candidates = [script_dir / "results.csv", Path("results.csv")]
        results_csv = next((str(p) for p in candidates if p.exists()), None)
    labels: dict[int, dict] = {}
    if results_csv and Path(results_csv).exists():
        for _, row in pd.read_csv(results_csv).iterrows():
            labels[int(row["episode_index"])] = {
                "success": bool(row.get("overall_success", False)),
                "pick":    str(row.get("pick_quality", "")),
                "drop":    str(row.get("drop_quality", "")),
            }
        print(f"Loaded {len(labels)} labels from {results_csv}")

    client           = genai.Client(api_key=load_api_key(key_file))
    buffer_size      = int(window_seconds * fps)
    frames_per_check = int(check_interval_s * fps)

    # Auto-detect camera key from meta columns (supports camera1, front, etc.)
    _cam_candidates = [
        "videos/observation.images.camera1",
        "videos/observation.images.front",
        "videos/observation.images.top",
        "videos/observation.images.camera_0",
    ]
    cam_key = next(
        (k for k in _cam_candidates if f"{k}/from_timestamp" in df_meta.columns),
        "videos/observation.images.camera1",  # fallback
    )
    print(f"  Using camera key: {cam_key}")
    chunk_col  = f"{cam_key}/chunk_index"
    file_col   = f"{cam_key}/file_index"
    from_col   = f"{cam_key}/from_timestamp"
    to_col     = f"{cam_key}/to_timestamp"

    episodes = sorted(df_data["episode_index"].unique())
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    rows: list[dict] = []
    episode_checks: dict[int, list[dict]] = {}   # {ep_idx: [{frame_idx, ts, p, ema, ...}]}
    video_paths: list[Path] = []

    for ep_idx in episodes:
        ep_df   = df_data[df_data["episode_index"] == ep_idx].reset_index(drop=True)
        ep_meta = df_meta[df_meta["episode_index"] == ep_idx]

        if ep_meta.empty or from_col not in ep_meta.columns:
            print(f"  [ep {ep_idx}] no meta — skipping")
            continue
        row_m    = ep_meta.iloc[0]
        from_ts  = float(row_m[from_col])
        to_ts    = float(row_m[to_col]) if to_col in ep_meta.columns else None
        # cam_key is "videos/observation.images.xxx" — strip the "videos/" prefix for path
        cam_subdir = cam_key.removeprefix("videos/")
        vid_path = (Path(repo_dir) / "videos" / cam_subdir
                    / f"chunk-{int(row_m[chunk_col]):03d}"
                    / f"file-{int(row_m[file_col]):03d}.mp4")
        if not vid_path.exists():
            print(f"  [ep {ep_idx}] video not found — skipping")
            continue

        cap        = cv2.VideoCapture(str(vid_path))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        cap.set(cv2.CAP_PROP_POS_MSEC, from_ts * 1000)

        frame_buf:  deque = deque(maxlen=buffer_size)
        action_buf: deque = deque(maxlen=buffer_size)
        state_buf:  deque = deque(maxlen=buffer_size)

        n_checks            = 0
        struggle_score      = 0.0
        peak_score          = 0.0
        first_interrupt_ts: float | None = None
        last_result         = _DEFAULT_INTERRUPT.copy()
        frame_counter       = 0
        next_check_at       = int(10.0 * actual_fps)  # first check after 10s
        all_probs: list[float] = []
        episode_checks[ep_idx] = []
        p_median_buf: list[float] = []   # rolling 3-check median filter

        n_ep_frames = len(ep_df)
        print(f"\n[ep {ep_idx:02d}] {n_ep_frames} frames  from={from_ts:.1f}s")

        for data_idx in range(n_ep_frames):
            ok, frame = cap.read()
            if not ok:
                break
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if to_ts is not None and pos_ms / 1000.0 > to_ts + 0.1:
                break

            frame_buf.append(frame)
            action = np.asarray(ep_df.at[data_idx, "action"],            dtype=np.float32)
            state  = np.asarray(ep_df.at[data_idx, "observation.state"], dtype=np.float32)
            action_buf.append(action)
            state_buf.append(state)
            frame_counter += 1

            if frame_counter >= next_check_at and len(frame_buf) >= n_sample_frames:
                next_check_at += frames_per_check
                frames_snap  = LiveStruggleMonitor._subsample(list(frame_buf), n_sample_frames)
                actions_snap = np.stack(list(action_buf)) if len(action_buf) >= 5 else None
                states_snap  = np.stack(list(state_buf))  if len(state_buf)  >= 5 else None
                gripper_snap = (
                    states_snap[:, gripper_col_idx]
                    if states_snap is not None else None
                )

                cur_ts = from_ts + frame_counter / actual_fps
                try:
                    result = assess_interrupt(
                        frames_snap, client, model,
                        actions=actions_snap, states=states_snap,
                        gripper=gripper_snap, gripper_threshold=gripper_threshold,
                    )
                    last_result   = result
                    n_checks     += 1
                    p_raw         = float(result["interrupt_probability"])
                    # Time ramp: scale down early-episode probabilities
                    ep_elapsed    = frame_counter / actual_fps  # seconds into this episode
                    time_factor   = min(1.0, ep_elapsed / TIME_RAMP_END_S)
                    # Rising cap: 0.75 early, grows to 0.90 after CAP_RAMP_END_S
                    cap_ramp      = min(1.0, max(0.0, ep_elapsed - TIME_RAMP_END_S)
                                        / max(1.0, CAP_RAMP_END_S - TIME_RAMP_END_S))
                    p_cap         = P_CAP_MIN + (P_CAP_MAX - P_CAP_MIN) * cap_ramp
                    p             = min(p_raw * time_factor, p_cap)
                    # Median filter: buffer last 3 scaled p values to kill single-check spikes
                    p_median_buf.append(p)
                    if len(p_median_buf) > 3:
                        p_median_buf.pop(0)
                    p_filtered    = float(np.median(p_median_buf))
                    struggle_score = ema_alpha * p_filtered + (1 - ema_alpha) * struggle_score
                    all_probs.append(p)
                    peak_score    = max(peak_score, struggle_score)
                    flagged       = struggle_score >= interrupt_threshold
                    if flagged and first_interrupt_ts is None:
                        first_interrupt_ts = cur_ts
                    episode_checks[ep_idx].append({
                        "frame_idx":   frame_counter,
                        "ts":          cur_ts,
                        "p":           p_filtered,
                        "p_raw":       p_raw,
                        "time_factor": round(time_factor, 2),
                        "p_cap":       round(p_cap, 2),
                        "ema":         struggle_score,
                        "picks":       result["pickup_attempts"],
                        "drops":       result["drop_attempts"],
                        "reason":      result["reason"],
                    })
                    status = "INTERRUPT" if flagged else "ok     "
                    print(
                        f"  t={cur_ts:.1f}s  {status}"
                        f"  p={p_filtered:.2f}(raw={p_raw:.2f}×{time_factor:.2f} cap={p_cap:.2f})"
                        f"  ema={struggle_score:.2f}"
                        f"  picks={result['pickup_attempts']}"
                        f"  drops={result['drop_attempts']}"
                        f"  | {result['reason']}"
                    )
                except Exception as e:
                    print(f"  t={cur_ts:.1f}s  Gemini error: {e}")

        cap.release()

        lab             = labels.get(ep_idx, {})
        final_interrupt = struggle_score >= interrupt_threshold
        mean_prob       = float(np.mean(all_probs)) if all_probs else 0.0

        rows.append({
            "episode_index":        ep_idx,
            "n_checks":             n_checks,
            "mean_interrupt_prob":  round(mean_prob, 3),
            "peak_ema_score":       round(peak_score, 3),
            "final_ema_score":      round(struggle_score, 3),
            "final_interrupt":      final_interrupt,
            "first_interrupt_ts":   first_interrupt_ts,
            "final_pickup_attempts": last_result["pickup_attempts"],
            "final_drop_attempts":   last_result["drop_attempts"],
            "final_reason":         last_result["reason"],
            "label_success":        lab.get("success", None),
            "label_pick":           lab.get("pick",    None),
            "label_drop":           lab.get("drop",    None),
        })
        print(
            f"  -> {'INTERRUPT' if final_interrupt else 'ok'}"
            f"  ema={struggle_score:.2f}  peak={peak_score:.2f}"
            f"  mean_p={mean_prob:.2f}"
            + (f"  label_success={lab.get('success')}" if lab else "")
        )

        # Render video for this episode immediately using already-loaded data
        if video_dir is not None and episode_checks.get(ep_idx):
            actions_full = np.stack(ep_df["action"].values).astype(np.float32)
            states_full  = np.stack(ep_df["observation.state"].values).astype(np.float32)
            gripper_full = states_full[:, gripper_col_idx]
            out_vid = Path(video_dir) / f"episode_{ep_idx:04d}_interrupt.mp4"
            _render_episode_interrupt_video(
                ep_idx=ep_idx,
                checks=episode_checks[ep_idx],
                actions=actions_full,
                states=states_full,
                gripper=gripper_full,
                vid_path=vid_path,
                from_ts=from_ts,
                to_ts=to_ts if to_ts is not None else from_ts + len(ep_df) / fps,
                out_path=out_vid,
                interrupt_threshold=interrupt_threshold,
                gripper_threshold=gripper_threshold,
            )
            video_paths.append(out_vid)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")

    if labels and "label_success" in results_df.columns:
        labeled = results_df.dropna(subset=["label_success"])
        failed  = labeled[labeled["label_success"] == False]
        success = labeled[labeled["label_success"] == True]

        sensitivity = (failed["final_interrupt"] == True).mean()  if len(failed)  else float("nan")
        specificity = (success["final_interrupt"] == False).mean() if len(success) else float("nan")

        # AUC-like: mean prob on failed vs success
        mean_p_fail = failed["mean_interrupt_prob"].mean()   if len(failed)  else float("nan")
        mean_p_succ = success["mean_interrupt_prob"].mean()  if len(success) else float("nan")

        print(f"\n=== Summary ===")
        print(f"  Episodes evaluated  : {len(results_df)}")
        print(f"  Labeled episodes    : {len(labeled)}  (failed={len(failed)}, success={len(success)})")
        print(f"  Sensitivity         : {sensitivity:.2%}")
        print(f"  Specificity         : {specificity:.2%}")
        print(f"  Mean p (failed)     : {mean_p_fail:.3f}")
        print(f"  Mean p (success)    : {mean_p_succ:.3f}")
        print(f"  Separation          : {mean_p_fail - mean_p_succ:.3f}  (higher = better)")

    if video_paths:
        from batch_detect_phases import concatenate_videos
        concat_out = Path(video_dir) / "all_interrupt.mp4"
        print(f"\nConcatenating {len(video_paths)} videos -> {concat_out}")
        concatenate_videos(video_paths, concat_out)
        print(f"Done: {concat_out}")

    return results_df


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Struggle monitor tools.")
    sub = parser.add_subparsers(dest="cmd")

    # ── live demo on a single video ──
    p_live = sub.add_parser("live", help="Simulate live monitoring on a single .mp4.")
    p_live.add_argument("video",      help="Path to an episode .mp4.")
    p_live.add_argument("--key-file", default=r"C:/Users/calle/Desktop/gem.txt")
    p_live.add_argument("--model",    default="gemini-2.5-flash")
    p_live.add_argument("--interval", type=float, default=2.0)
    p_live.add_argument("--threshold", type=float, default=0.6)

    # ── batch dataset test ──
    p_batch = sub.add_parser("batch", help="Run batch test on a full LeRobot dataset.")
    p_batch.add_argument("--dataset",    default="nc8304/eval_smolvla-phase-split_combined")
    p_batch.add_argument("--key-file",   default=r"C:/Users/calle/Desktop/gem.txt")
    p_batch.add_argument("--model",      default="gemini-2.5-flash")
    p_batch.add_argument("--interval",   type=float, default=2.0)
    p_batch.add_argument("--window",     type=float, default=20.0, help="Rolling buffer seconds")
    p_batch.add_argument("--threshold",  type=float, default=0.5,  help="EMA interrupt threshold")
    p_batch.add_argument("--ema-alpha",  type=float, default=0.25, help="EMA smoothing factor")
    p_batch.add_argument("--results-csv", default=None)
    p_batch.add_argument("--out-csv",    default="struggle_test_results.csv")
    p_batch.add_argument("--video-dir",    default=None,
                         help="If set, render 3-panel interrupt videos here.")
    p_batch.add_argument("--max-episodes", type=int, default=None,
                         help="Stop after this many episodes.")

    args = parser.parse_args()

    if args.cmd == "live" or args.cmd is None:
        # Backwards compat: treat bare positional as live mode
        video_path = getattr(args, "video", None)
        if video_path is None:
            parser.print_help()
            raise SystemExit(1)

        monitor = LiveStruggleMonitor(
            key_file=args.key_file,
            model=args.model,
            check_interval=args.interval,
            struggle_threshold=args.threshold,
        )
        monitor.start()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"Simulating live feed from {video_path}  ({fps:.0f} fps)\n")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            monitor.push_frame(frame)
            time.sleep(1.0 / fps)

        cap.release()
        monitor.stop()

    elif args.cmd == "batch":
        test_on_dataset(
            dataset_id=args.dataset,
            key_file=args.key_file,
            model=args.model,
            check_interval_s=args.interval,
            window_seconds=args.window,
            interrupt_threshold=args.threshold,
            ema_alpha=args.ema_alpha,
            results_csv=args.results_csv,
            out_csv=args.out_csv,
            video_dir=args.video_dir,
            max_episodes=args.max_episodes,
        )
