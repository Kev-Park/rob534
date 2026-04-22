"""Interactive frame scrubber to label pickup / hold / drop phase boundaries.

Loads a rollout video, lets you scrub through a frame range, and mark the
exact frame where each phase transition occurs.  Outputs the labelled frames
as JSON so they can be used later for automated detection.

Usage:
    python phase_scrubber.py rollouts/rollout_20260419_124105.mp4
    python phase_scrubber.py rollouts/rollout_20260419_124105.mp4 --start 60 --end 180

Controls:
    LEFT / RIGHT arrow   step 1 frame
    , / .                step 10 frames
    SPACE                play / pause
    1                    mark PICKUP frame  (robot just grabbed block)
    2                    mark HOLD   frame  (reference "holding" state)
    3                    mark DROP   frame  (robot just released block)
    u                    undo last mark
    s / ENTER            save labels to JSON and exit
    q / ESC              quit without saving
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PHASE_KEYS = {
    ord("1"): "pickup",
    ord("2"): "hold",
    ord("3"): "drop",
}
PHASE_COLORS = {
    "pickup": (0, 255, 128),   # green
    "hold":   (0, 200, 255),   # yellow
    "drop":   (80, 80, 255),   # red
}


def load_frames(video_path: Path, start: int, end: int | None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(end if end is not None else total, total)
    start = max(0, start)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(end - start):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames [{start}–{start + len(frames) - 1}]")
    return frames, start


def draw_overlay(frame: np.ndarray, abs_idx: int, labels: dict, fps: float) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]

    # frame / time info
    ts = abs_idx / fps
    cv2.putText(img, f"frame {abs_idx}  |  {ts:.2f}s",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # show any marks on this exact frame
    for phase, fidx in labels.items():
        if fidx == abs_idx:
            color = PHASE_COLORS[phase]
            cv2.putText(img, f"[{phase.upper()}]",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)

    # legend at bottom
    y = h - 10
    legend = "1=PICKUP  2=HOLD  3=DROP  u=undo  s=save  q=quit  SPACE=play"
    cv2.putText(img, legend, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)

    # marker strip along top edge
    bar_w = w - 20
    for phase, fidx in labels.items():
        if fidx is None:
            continue
        color = PHASE_COLORS[phase]
        cv2.circle(img, (10, 8), 6, color, -1)   # just a dot per phase in corner
        # tick on a timeline bar
        pct = (fidx - abs_idx) / max(len(labels), 1)   # rough position

    return img


def run(video_path: Path, start: int, end: int | None, out_json: Path):
    cap_tmp = cv2.VideoCapture(str(video_path))
    fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 30.0
    cap_tmp.release()

    frames, frame_offset = load_frames(video_path, start, end)
    if not frames:
        print("No frames loaded — check --start / --end range.")
        sys.exit(1)

    n = len(frames)
    idx = 0
    labels: dict[str, int | None] = {"pickup": None, "hold": None, "drop": None}
    playing = False

    win = "Phase Scrubber"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 560)
    # must imshow before creating trackbar on Windows
    cv2.imshow(win, frames[0])
    cv2.waitKeyEx(1)
    cv2.createTrackbar("frame", win, 0, n - 1, lambda _: None)

    while True:
        # exit gracefully if user closes the window
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed.")
            break

        # trackbar is authoritative when not playing
        if not playing:
            idx = cv2.getTrackbarPos("frame", win)

        idx = max(0, min(idx, n - 1))
        abs_idx = frame_offset + idx
        img = draw_overlay(frames[idx], abs_idx, labels, fps)

        # coloured phase markers along bottom of image
        h, w = img.shape[:2]
        for phase, fidx in labels.items():
            if fidx is None:
                continue
            rel = fidx - frame_offset
            x = int(10 + (w - 20) * rel / max(n - 1, 1))
            color = PHASE_COLORS[phase]
            cv2.circle(img, (x, h - 25), 5, color, -1)
            cv2.putText(img, phase[0].upper(), (x - 4, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imshow(win, img)

        # waitKeyEx gives proper arrow-key codes on Windows
        key = cv2.waitKeyEx(33 if playing else 15)

        if playing:
            new_idx = min(idx + 1, n - 1)
            cv2.setTrackbarPos("frame", win, new_idx)
            if new_idx == n - 1:
                playing = False
            continue

        # --- key handling ---
        if key == -1:
            continue
        elif key in (ord("q"), 27):          # q / ESC
            print("Quit without saving.")
            break
        elif key in (ord("s"), 13):          # s / ENTER
            _save(labels, fps, frame_offset, out_json)
            break
        elif key == ord(" "):
            playing = True
        elif key in (2424832, 65361, 81):    # LEFT arrow
            cv2.setTrackbarPos("frame", win, max(idx - 1, 0))
        elif key in (2555904, 65363, 83):    # RIGHT arrow
            cv2.setTrackbarPos("frame", win, min(idx + 1, n - 1))
        elif key == ord(","):
            cv2.setTrackbarPos("frame", win, max(idx - 10, 0))
        elif key == ord("."):
            cv2.setTrackbarPos("frame", win, min(idx + 10, n - 1))
        elif key in PHASE_KEYS:
            phase = PHASE_KEYS[key]
            labels[phase] = abs_idx
            print(f"  Marked {phase:7s} @ frame {abs_idx}  ({abs_idx/fps:.2f}s)")
        elif key == ord("u"):
            set_phases = [(p, f) for p, f in labels.items() if f is not None]
            if set_phases:
                last = max(set_phases, key=lambda x: x[1])
                labels[last[0]] = None
                print(f"  Cleared {last[0]}")

    cv2.destroyAllWindows()


def _save(labels: dict, fps: float, frame_offset: int, out_json: Path):
    missing = [p for p, f in labels.items() if f is None]
    if missing:
        print(f"  Warning: not all phases marked — missing: {missing}")

    data = {
        phase: {
            "frame": fidx,
            "time_s": round(fidx / fps, 3) if fidx is not None else None,
        }
        for phase, fidx in labels.items()
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved labels → {out_json}")
    print(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Label pickup/hold/drop phase frames")
    parser.add_argument("video", nargs="?", default=None,
                        help="Path to rollout video (default: newest in rollouts/)")
    parser.add_argument("--start", type=int, default=0,
                        help="First frame to load (default: 0)")
    parser.add_argument("--end",   type=int, default=None,
                        help="Last frame to load exclusive (default: end of video)")
    parser.add_argument("--out",   type=str, default=None,
                        help="Output JSON path (default: <video_stem>_phases.json)")
    args = parser.parse_args()

    if args.video:
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

    out_json = Path(args.out) if args.out else video_path.with_name(video_path.stem + "_phases.json")

    print(f"Video : {video_path}")
    print(f"Range : frames {args.start} – {args.end or 'end'}")
    print(f"Output: {out_json}")
    print()

    run(video_path, args.start, args.end, out_json)


if __name__ == "__main__":
    main()
