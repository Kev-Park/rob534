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

# ── Prompt ────────────────────────────────────────────────────────────────────

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

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "struggling": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason":     {"type": "string"},
    },
    "required": ["struggling", "confidence", "reason"],
}

# ── Default signal (returned before the first Gemini call completes) ──────────

_DEFAULT_SIGNAL = {"struggling": False, "confidence": 0.0, "reason": "no assessment yet"}


# ── Core assessment function ──────────────────────────────────────────────────

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


# ── Background monitor ────────────────────────────────────────────────────────

class LiveStruggleMonitor:
    """Continuously watches a rolling frame buffer and polls Gemini for struggle signals.

    The Gemini call runs in a background thread so your control loop is
    never blocked. The latest signal is always available via is_struggling().

    Args:
        key_file:       Path to Gemini API key file (or None to use env var).
        model:          Gemini model to use.
        check_interval: How often (seconds) to send frames to Gemini.
        buffer_seconds: How many seconds of frames to keep in the rolling buffer.
        n_sample_frames: How many frames to subsample from the buffer per check.
        fps:            Expected camera frame rate (used for buffer sizing).
        struggle_threshold: Confidence threshold above which is_struggling() → True.
    """

    def __init__(
        self,
        key_file: str | None = None,
        model: str = "gemini-2.5-flash",
        check_interval: float = 2.0,
        buffer_seconds: float = 3.0,
        n_sample_frames: int = 8,
        fps: float = 30.0,
        struggle_threshold: float = 0.6,
    ):
        self._client           = genai.Client(api_key=load_api_key(key_file))
        self._model            = model
        self._check_interval   = check_interval
        self._n_sample         = n_sample_frames
        self._threshold        = struggle_threshold

        buffer_size              = int(buffer_seconds * fps)
        self._buffer: deque      = deque(maxlen=buffer_size)
        self._action_buf: deque  = deque(maxlen=buffer_size)
        self._state_buf:  deque  = deque(maxlen=buffer_size)
        self._lock               = threading.Lock()

        self._signal: dict     = _DEFAULT_SIGNAL.copy()
        self._stop_event       = threading.Event()
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
        """True if the latest Gemini assessment says the policy is struggling."""
        sig = self._signal
        return bool(sig["struggling"]) and sig["confidence"] >= self._threshold

    def get_signal(self) -> dict:
        """Return the full latest signal: {struggling, confidence, reason}."""
        return self._signal.copy()

    def reset_signal(self) -> None:
        """Clear the current signal (call at the start of each new episode)."""
        self._signal = _DEFAULT_SIGNAL.copy()

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
                buf = list(self._buffer)
                actions = (
                    np.stack(list(self._action_buf))
                    if len(self._action_buf) >= 5 else None
                )
                states = (
                    np.stack(list(self._state_buf))
                    if len(self._state_buf) >= 5 else None
                )

            if len(buf) >= self._n_sample:
                frames = self._subsample(buf, self._n_sample)
                try:
                    signal = assess_frames(
                        frames, self._client, self._model,
                        actions=actions, states=states,
                    )
                    self._signal = signal
                    status = "STRUGGLING" if self.is_struggling() else "ok"
                    print(
                        f"[StruggleMonitor] {status}"
                        f"  conf={signal['confidence']:.2f}"
                        f"  | {signal['reason']}"
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


# ── Batch dataset test ────────────────────────────────────────────────────────

def test_on_dataset(
    dataset_id: str = "nc8304/eval_smolvla-phase-split_combined",
    key_file: str | None = None,
    model: str = "gemini-2.5-flash",
    check_interval_s: float = 2.0,
    buffer_seconds: float = 3.0,
    n_sample_frames: int = 8,
    fps: float = 30.0,
    struggle_threshold: float = 0.6,
    results_csv: str | None = None,
    out_csv: str = "struggle_test_results.csv",
) -> pd.DataFrame:
    """Run the struggle monitor over every episode in a LeRobot dataset and compare
    to known labels.

    The function simulates the live monitor: it steps through each episode's frames
    in real-index order, maintains rolling action/state deques, and calls assess_frames
    at ``check_interval_s`` intervals (measured in video time, not wall time).

    Args:
        dataset_id:        HuggingFace dataset ID.
        key_file:          Path to Gemini API key (or None for env var).
        model:             Gemini model to use.
        check_interval_s:  How many seconds of video time between Gemini calls.
        buffer_seconds:    Rolling buffer length in video seconds.
        n_sample_frames:   Frames subsampled per Gemini call.
        fps:               Assumed camera frame rate for buffer sizing.
        struggle_threshold: Confidence cutoff for is_struggling.
        results_csv:       Path to the labeled results.csv (auto-detected if None).
        out_csv:           Where to write the per-episode comparison CSV.

    Returns:
        DataFrame with columns: episode_index, n_checks, n_struggling,
        first_struggle_ts, final_struggling, final_confidence, final_reason,
        label_success (from results_csv), label_pick, label_drop.
    """
    # ── load dataset ──────────────────────────────────────────────────────────
    print(f"Downloading dataset {dataset_id} ...")
    repo_dir = snapshot_download(repo_id=dataset_id, repo_type="dataset")

    data_files = sorted(glob.glob(f"{repo_dir}/data/**/*.parquet", recursive=True))
    meta_files = sorted(glob.glob(f"{repo_dir}/meta/episodes/**/*.parquet", recursive=True))

    df_data = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    df_meta = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)

    # ── load known labels ─────────────────────────────────────────────────────
    if results_csv is None:
        script_dir = Path(__file__).parent
        candidates = [script_dir / "results.csv", Path("results.csv")]
        results_csv = next((str(p) for p in candidates if p.exists()), None)

    labels: dict[int, dict] = {}
    if results_csv and Path(results_csv).exists():
        ldf = pd.read_csv(results_csv)
        for _, row in ldf.iterrows():
            labels[int(row["episode_index"])] = {
                "success":  bool(row.get("overall_success", False)),
                "pick":     str(row.get("pick_quality", "")),
                "drop":     str(row.get("drop_quality", "")),
            }
        print(f"Loaded {len(labels)} labels from {results_csv}")
    else:
        print("No labels CSV found — will omit label columns")

    # ── Gemini client ─────────────────────────────────────────────────────────
    client = genai.Client(api_key=load_api_key(key_file))
    buffer_size = int(buffer_seconds * fps)
    frames_per_check = int(check_interval_s * fps)

    episodes = sorted(df_data["episode_index"].unique())
    rows = []

    for ep_idx in episodes:
        ep_df   = df_data[df_data["episode_index"] == ep_idx].reset_index(drop=True)
        ep_meta = df_meta[df_meta["episode_index"] == ep_idx]

        # Locate video clip
        vid_ts_col = "videos/observation.images.camera1/from_timestamp"
        chunk_col  = "videos/observation.images.camera1/chunk_index"
        file_col   = "videos/observation.images.camera1/file_index"

        if not ep_meta.empty and vid_ts_col in ep_meta.columns:
            row_m       = ep_meta.iloc[0]
            chunk_idx   = int(row_m[chunk_col])
            file_idx    = int(row_m[file_col])
            from_ts     = float(row_m[vid_ts_col])
            to_ts_col   = "videos/observation.images.camera1/to_timestamp"
            to_ts       = float(row_m[to_ts_col]) if to_ts_col in ep_meta.columns else None
            vid_path    = Path(repo_dir) / "videos" / "observation.images.camera1" \
                          / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
        else:
            print(f"  [ep {ep_idx}] could not locate video — skipping")
            continue

        if not vid_path.exists():
            print(f"  [ep {ep_idx}] video not found: {vid_path} — skipping")
            continue

        # Open video and seek to episode start
        cap = cv2.VideoCapture(str(vid_path))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        cap.set(cv2.CAP_PROP_POS_MSEC, from_ts * 1000)

        frame_buf:  deque = deque(maxlen=buffer_size)
        action_buf: deque = deque(maxlen=buffer_size)
        state_buf:  deque = deque(maxlen=buffer_size)

        n_checks       = 0
        n_struggling   = 0
        first_struggle_ts: float | None = None
        last_signal    = _DEFAULT_SIGNAL.copy()
        frame_counter  = 0
        next_check_at  = frames_per_check   # frame index (within episode) for next Gemini call

        n_ep_frames = len(ep_df)
        print(f"\n[ep {ep_idx:02d}] {n_ep_frames} frames  video={vid_path.name}  from={from_ts:.1f}s")

        for data_idx in range(n_ep_frames):
            ok, frame = cap.read()
            if not ok:
                break

            # Stop if we've passed the episode's end timestamp
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if to_ts is not None and pos_ms / 1000.0 > to_ts + 0.1:
                break

            frame_buf.append(frame)

            action = np.asarray(ep_df.at[data_idx, "action"], dtype=np.float32)
            state  = np.asarray(ep_df.at[data_idx, "observation.state"], dtype=np.float32)
            action_buf.append(action)
            state_buf.append(state)

            frame_counter += 1

            if frame_counter >= next_check_at and len(frame_buf) >= n_sample_frames:
                next_check_at += frames_per_check
                frames_snap   = LiveStruggleMonitor._subsample(list(frame_buf), n_sample_frames)
                actions_snap  = (
                    np.stack(list(action_buf)) if len(action_buf) >= 5 else None
                )
                states_snap   = (
                    np.stack(list(state_buf)) if len(state_buf) >= 5 else None
                )

                cur_ts = from_ts + frame_counter / actual_fps
                try:
                    signal = assess_frames(
                        frames_snap, client, model,
                        actions=actions_snap, states=states_snap,
                    )
                    last_signal = signal
                    n_checks += 1
                    is_str = bool(signal["struggling"]) and signal["confidence"] >= struggle_threshold
                    if is_str:
                        n_struggling += 1
                        if first_struggle_ts is None:
                            first_struggle_ts = cur_ts
                    status = "STRUGGLING" if is_str else "ok      "
                    print(
                        f"  t={cur_ts:.1f}s  {status}"
                        f"  conf={signal['confidence']:.2f}"
                        f"  | {signal['reason']}"
                    )
                except Exception as e:
                    print(f"  t={cur_ts:.1f}s  Gemini error: {e}")

        cap.release()

        lab = labels.get(ep_idx, {})
        final_struggling = (
            bool(last_signal["struggling"])
            and last_signal["confidence"] >= struggle_threshold
        )
        rows.append({
            "episode_index":      ep_idx,
            "n_checks":           n_checks,
            "n_struggling":       n_struggling,
            "first_struggle_ts":  first_struggle_ts,
            "final_struggling":   final_struggling,
            "final_confidence":   last_signal["confidence"],
            "final_reason":       last_signal["reason"],
            "label_success":      lab.get("success", None),
            "label_pick":         lab.get("pick",    None),
            "label_drop":         lab.get("drop",    None),
        })
        print(
            f"  -> final={'STRUGGLING' if final_struggling else 'ok'}"
            f"  n_checks={n_checks}  n_struggling={n_struggling}"
            + (f"  label_success={lab.get('success')}" if lab else "")
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if labels:
        # Agreement: label_success=False -> we should flag as struggling at some point
        failed_eps  = results_df[results_df["label_success"] == False]
        success_eps = results_df[results_df["label_success"] == True]

        sensitivity = (
            (failed_eps["n_struggling"] > 0).mean()
            if len(failed_eps) else float("nan")
        )
        specificity = (
            (success_eps["n_struggling"] == 0).mean()
            if len(success_eps) else float("nan")
        )
        print(f"\n=== Summary ===")
        print(f"  Episodes evaluated : {len(results_df)}")
        print(f"  Failed episodes    : {len(failed_eps)}")
        print(f"  Sensitivity (failed -> flagged)  : {sensitivity:.2%}")
        print(f"  Specificity (success -> not flagged): {specificity:.2%}")

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
    p_batch.add_argument("--dataset",   default="nc8304/eval_smolvla-phase-split_combined")
    p_batch.add_argument("--key-file",  default=r"C:/Users/calle/Desktop/gem.txt")
    p_batch.add_argument("--model",     default="gemini-2.5-flash")
    p_batch.add_argument("--interval",  type=float, default=2.0)
    p_batch.add_argument("--threshold", type=float, default=0.6)
    p_batch.add_argument("--results-csv", default=None, help="Path to labeled results.csv")
    p_batch.add_argument("--out-csv",   default="struggle_test_results.csv")

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
            struggle_threshold=args.threshold,
            results_csv=args.results_csv,
            out_csv=args.out_csv,
        )
