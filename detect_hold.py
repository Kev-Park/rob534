"""Detect the 'hold' phase by finding segments where the bottom ROI is stable.

When the robot is holding the block the bottom of the frame barely changes
between consecutive frames.  This script:
  1. Computes per-frame MAD (mean absolute diff) in a configurable bottom ROI
  2. Plots the diff curve so you can see pickup / hold / drop transitions
  3. Detects the longest stable segment (hold candidate)
  4. Saves a phases JSON compatible with phase_scrubber.py

Usage:
    python detect_hold.py rollouts/rollout_20260419_124609.mp4
    python detect_hold.py rollouts/rollout_20260419_124609.mp4 --roi_frac 0.35 --threshold 8
    python detect_hold.py rollouts/rollout_20260419_124609.mp4 --show_roi   # preview ROI crop
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np


def compute_diff_curve(video_path: Path, roi_frac: float,
                       start: int = 0, end: int | None = None) -> tuple[np.ndarray, float, int]:
    """Return (per_frame_mad, fps, total_frames) for the given frame range."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end is None:
        end = total
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    diffs = []
    prev_roi = None
    for _ in range(end - start):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        roi = gray[int(h * (1 - roi_frac)):, :]
        if prev_roi is not None:
            mad = np.mean(np.abs(roi.astype(np.float32) - prev_roi.astype(np.float32)))
            diffs.append(mad)
        else:
            diffs.append(0.0)
        prev_roi = roi
    cap.release()
    return np.array(diffs), fps, total


def find_stable_segments(diffs: np.ndarray, threshold: float,
                          min_len: int = 10) -> list[tuple[int, int]]:
    """Return list of (start, end) frame indices where diff < threshold."""
    stable = diffs < threshold
    segments = []
    in_seg = False
    seg_start = 0
    for i, s in enumerate(stable):
        if s and not in_seg:
            seg_start = i
            in_seg = True
        elif not s and in_seg:
            if i - seg_start >= min_len:
                segments.append((seg_start, i - 1))
            in_seg = False
    if in_seg and len(diffs) - seg_start >= min_len:
        segments.append((seg_start, len(diffs) - 1))
    return segments


def show_roi_preview(video_path: Path, roi_frac: float):
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return
    h, w = frame.shape[:2]
    roi_y = int(h * (1 - roi_frac))
    preview = frame.copy()
    cv2.rectangle(preview, (0, roi_y), (w, h), (0, 255, 0), 3)
    cv2.putText(preview, f"ROI (bottom {roi_frac*100:.0f}%)",
                (10, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.namedWindow("ROI Preview", cv2.WINDOW_NORMAL)
    cv2.imshow("ROI Preview", preview)
    print("ROI preview — press any key to continue.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def build_plot_image(diffs: np.ndarray, fps: float, threshold: float,
                     segments: list[tuple[int, int]], video_name: str,
                     frame_w: int, plot_h: int = 180) -> tuple:
    """Render the base plot (no dot) to a BGR numpy image sized (plot_h, frame_w).
    Returns (bgr_image, ax, fig) so callers can compute data->pixel transforms.
    """
    dpi = 100
    fig_w = frame_w / dpi
    fig_h = plot_h / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    times = np.arange(len(diffs)) / fps
    ax.plot(times, diffs, color="steelblue", linewidth=0.8, label="bottom-ROI MAD")
    ax.axhline(threshold, color="tomato", linestyle="--", linewidth=1.0,
               label=f"threshold={threshold:.1f}")

    for i, (s, e) in enumerate(segments):
        ax.axvspan(s / fps, e / fps, alpha=0.25, color="lime",
                   label="stable" if i == 0 else "")

    if segments:
        longest = max(segments, key=lambda x: x[1] - x[0])
        mid = (longest[0] + longest[1]) / 2 / fps
        ax.text(mid, diffs.max() * 0.85, "HOLD", ha="center",
                fontsize=8, color="darkgreen", fontweight="bold")

    ax.set_xlim(0, (len(diffs) - 1) / fps)
    ax.set_ylim(0, diffs.max() * 1.1)
    ax.set_ylabel("MAD", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout(pad=0.3)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

    # ensure exact width
    if bgr.shape[1] != frame_w:
        bgr = cv2.resize(bgr, (frame_w, plot_h))

    return bgr, ax, fig


def render_annotated_video(video_path: Path, diffs: np.ndarray, fps: float,
                            threshold: float, segments: list[tuple[int, int]],
                            out_path: Path, plot_h: int = 180,
                            start: int = 0, end: int | None = None):
    cap = cv2.VideoCapture(str(video_path))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end is None:
        end = total
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    base_plot, ax, fig = build_plot_image(
        diffs, fps, threshold, segments, video_path.name, frame_w, plot_h)

    # precompute x-pixel position for each frame's time value
    total_time = (len(diffs) - 1) / fps
    def time_to_x(t):
        # ax.transData maps data coords -> figure pixels (origin bottom-left)
        disp = ax.transData.transform((t, 0))
        return int(disp[0])

    out_h = frame_h + plot_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, out_h))

    from tqdm import tqdm
    for fi in tqdm(range(end - start), desc="Rendering"):
        ret, frame = cap.read()
        if not ret:
            break

        # copy base plot and draw dot + vertical line at current time
        overlay = base_plot.copy()
        t = fi / fps
        x = time_to_x(t)
        x = max(1, min(x, frame_w - 2))
        cv2.line(overlay, (x, 0), (x, plot_h), (255, 255, 0), 1)
        y_dot = plot_h // 2
        if fi < len(diffs):
            # map diff value to y pixel
            disp = ax.transData.transform((t, float(diffs[fi])))
            y_dot = plot_h - int(disp[1])
            y_dot = max(4, min(y_dot, plot_h - 4))
        cv2.circle(overlay, (x, y_dot), 5, (0, 255, 255), -1)

        combined = np.vstack([overlay, frame])
        writer.write(combined)

    cap.release()
    writer.release()
    plt.close(fig)
    print(f"Video saved -> {out_path}")


def plot_curve(diffs: np.ndarray, fps: float, threshold: float,
               segments: list[tuple[int, int]], video_name: str, out_json: Path):
    base, ax, fig = build_plot_image(diffs, fps, threshold, segments, video_name, 1280, 400)
    png_path = out_json.with_suffix(".png")
    cv2.imwrite(str(png_path), base)
    print(f"Plot saved -> {png_path}")
    plt.close(fig)
    import os
    os.startfile(str(png_path))


ORIG_ROOT = Path(r"C:\Users\calle\.cache\huggingface\lerobot\nc8304\so101_combined")
AUG_ROOT  = Path(__file__).parent.parent / "outputs" / "augmented_datasets"
VIDEO_KEY = "observation.images.front"


def episode_frame_range(episode_index: int, variant: str = "original") -> tuple[Path, int, int]:
    """Return (video_path, start_frame, end_frame) for a given episode."""
    if variant == "original":
        root = ORIG_ROOT
    else:
        root = AUG_ROOT / variant
    ep = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    row = ep[ep["episode_index"] == episode_index].iloc[0]
    file_idx  = int(row[f"videos/{VIDEO_KEY}/file_index"])
    from_ts   = float(row[f"videos/{VIDEO_KEY}/from_timestamp"])
    to_ts     = float(row[f"videos/{VIDEO_KEY}/to_timestamp"])
    fps       = 30.0
    start     = int(from_ts * fps)
    end       = int(to_ts   * fps)
    video_path = root / "videos" / VIDEO_KEY / "chunk-000" / f"file-{file_idx:03d}.mp4"
    return video_path, start, end


def main():
    parser = argparse.ArgumentParser(description="Detect hold phase from bottom-ROI stability")
    parser.add_argument("video", nargs="?", default=None,
                        help="Path to video, OR omit and use --episode")
    parser.add_argument("--episode", type=int, default=None,
                        help="Episode index from the dataset (uses original dataset)")
    parser.add_argument("--variant", type=str, default="original",
                        help="Dataset variant name, e.g. 'blue_a05' (default: original)")
    parser.add_argument("--start",     type=int,   default=0)
    parser.add_argument("--end",       type=int,   default=None)
    parser.add_argument("--roi_frac",  type=float, default=0.30,
                        help="Fraction of frame height to analyse from bottom (default 0.30)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="MAD threshold below which frame is 'stable' (auto if omitted)")
    parser.add_argument("--min_len",   type=int,   default=15,
                        help="Minimum frames for a stable segment (default 15)")
    parser.add_argument("--show_roi",   action="store_true",
                        help="Show a preview of the ROI crop and exit")
    parser.add_argument("--make_video", action="store_true",
                        help="Render annotated output video with diff curve overlay")
    parser.add_argument("--out",        type=str,   default=None)
    args = parser.parse_args()

    start_frame = args.start
    end_frame   = args.end

    if args.episode is not None:
        video_path, start_frame, end_frame = episode_frame_range(args.episode, args.variant)
        print(f"Episode {args.episode}: frames {start_frame}-{end_frame}  ({video_path.name})")
    elif args.video:
        video_path = Path(args.video)
    else:
        folder = Path("rollouts")
        videos = sorted(folder.glob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not videos:
            print("No .mp4 files in rollouts/")
            sys.exit(1)
        video_path = videos[0]

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    if args.show_roi:
        show_roi_preview(video_path, args.roi_frac)
        return

    ep_tag = f"_ep{args.episode}" if args.episode is not None else ""
    out_json = Path(args.out) if args.out else \
        video_path.with_name(video_path.stem + ep_tag + "_phases.json")

    print(f"Video    : {video_path}")
    print(f"Frames   : {start_frame} - {end_frame or 'end'}")
    print(f"ROI      : bottom {args.roi_frac*100:.0f}% of frame")
    print(f"Computing frame diffs...", end=" ", flush=True)
    diffs, fps, total = compute_diff_curve(video_path, args.roi_frac,
                                           start=start_frame, end=end_frame)
    print(f"done  ({total} frames @ {fps:.0f}fps)")

    # auto threshold: midpoint between median and 75th percentile of non-zero diffs
    threshold = args.threshold
    if threshold is None:
        nz = diffs[diffs > 0]
        threshold = float(np.percentile(nz, 25))
        print(f"Auto threshold: {threshold:.2f}  (use --threshold to override)")

    segments = find_stable_segments(diffs, threshold, args.min_len)
    print(f"Stable segments found: {len(segments)}")
    for s, e in segments:
        print(f"  frames {s:4d}–{e:4d}  ({s/fps:.2f}s – {e/fps:.2f}s)  len={e-s+1}")

    # pick longest as hold
    hold_start = hold_end = None
    if segments:
        s, e = max(segments, key=lambda x: x[1] - x[0])
        hold_start, hold_end = int(s), int(e)
        print(f"\nHold candidate: frames {hold_start}–{hold_end}  ({hold_start/fps:.2f}s – {hold_end/fps:.2f}s)")

    # save phases JSON
    data = {
        "pickup": {"frame": hold_start, "time_s": round(hold_start / fps, 3)} if hold_start else None,
        "hold":   {"frame": (hold_start + hold_end) // 2,
                   "time_s": round(((hold_start + hold_end) // 2) / fps, 3)} if hold_start else None,
        "drop":   {"frame": hold_end, "time_s": round(hold_end / fps, 3)} if hold_end else None,
        "roi_frac": args.roi_frac,
        "threshold": threshold,
    }
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Phases saved -> {out_json}")

    plot_curve(diffs, fps, threshold, segments, video_path.name, out_json)

    if args.make_video:
        vid_out = video_path.with_name(video_path.stem + ep_tag + "_annotated.mp4")
        render_annotated_video(video_path, diffs, fps, threshold, segments, vid_out,
                               start=start_frame, end=end_frame)
        import os
        os.startfile(str(vid_out))


if __name__ == "__main__":
    main()
