"""
Phase viewer for standalone rollout videos.

Scans rollouts/ for *.mp4 files that have a matching *_phases.json.
Shows PICKUP | HOLD | DROP frames side-by-side with an Episode trackbar.

Usage:
    python rollout_phase_viewer.py
    python rollout_phase_viewer.py --dir path/to/rollouts

Controls:
    Drag "Rollout" trackbar   jump to any rollout
    LEFT / RIGHT              ±1 rollout
    q / Esc                   quit
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────
DEFAULT_DIR = Path(__file__).parent / "rollouts"

THUMB_W, THUMB_H = 426, 320
GAP              = 6
LABEL_H          = 32
HEADER_H         = 26
STRIP_W          = THUMB_W * 3 + GAP * 2
STRIP_H          = HEADER_H + THUMB_H + LABEL_H

C_PICKUP = (0,   210,  60)
C_HOLD   = (0,   200, 200)
C_DROP   = (60,   60, 220)
LABELS   = ["PICKUP", "HOLD", "DROP"]
COLORS   = [C_PICKUP, C_HOLD, C_DROP]


# ── frame decoding ────────────────────────────────────────────────────────────

def decode_frames(vid_path: Path, frame_nums: list[int]) -> dict[int, np.ndarray]:
    """Decode specific frame numbers from a standalone MP4 via cv2."""
    targets = sorted(set(f for f in frame_nums if f is not None))
    captured = {}
    if not targets:
        return captured

    cap = cv2.VideoCapture(str(vid_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for fn in targets:
        if fn >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if ret:
            captured[fn] = cv2.resize(frame, (THUMB_W, THUMB_H))

    cap.release()
    return captured


# ── composite builder ─────────────────────────────────────────────────────────

def make_strip(name: str, pickup: int | None, hold: int | None,
               drop: int | None, fps: float, decoded: dict) -> np.ndarray:
    """Build one STRIP_H × STRIP_W BGR image for this rollout."""

    # header
    header = np.full((HEADER_H, STRIP_W, 3), 18, dtype=np.uint8)
    hold_str = ""
    if pickup is not None and drop is not None:
        hold_f = drop - pickup
        hold_str = f"hold {hold_f}f  ({hold_f / fps:.1f}s)"
    cv2.putText(header, f"{name}   {hold_str}",
                (8, HEADER_H - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (230, 230, 230), 1, cv2.LINE_AA)

    panels = []
    for frame_no, label, color in zip([pickup, hold, drop], LABELS, COLORS):
        img = decoded.get(frame_no)
        if img is None:
            thumb = np.full((THUMB_H, THUMB_W, 3), 35, dtype=np.uint8)
            cv2.putText(thumb, "N/A", (THUMB_W // 2 - 28, THUMB_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (110, 110, 110), 2)
        else:
            thumb = img.copy()
        cv2.rectangle(thumb, (0, 0), (THUMB_W - 1, THUMB_H - 1), color, 4)

        bar = np.full((LABEL_H, THUMB_W, 3), 22, dtype=np.uint8)
        frame_str = str(frame_no) if frame_no is not None else "?"
        txt = f"{label}   f {frame_str}"
        cv2.putText(bar, txt, (8, LABEL_H - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)

        panels.append(np.vstack([thumb, bar]))

    gap = np.full((THUMB_H + LABEL_H, GAP, 3), 8, dtype=np.uint8)
    row = np.hstack([panels[0], gap, panels[1], gap, panels[2]])
    return np.vstack([header, row])


# ── build strips ──────────────────────────────────────────────────────────────

def build_strips(rollout_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    """Find all rollouts with phase JSONs and build strips."""
    entries = sorted(rollout_dir.glob("*.mp4"))
    pairs = []
    for mp4 in entries:
        json_path = mp4.with_name(mp4.stem + "_phases.json")
        if json_path.exists():
            pairs.append((mp4, json_path))

    if not pairs:
        print(f"No rollouts with phase JSONs found in {rollout_dir}")
        return [], []

    strips = []
    names  = []
    n = len(pairs)

    for i, (mp4, json_path) in enumerate(pairs):
        with open(json_path) as f:
            phases = json.load(f)

        pickup = phases.get("pickup", {}).get("frame")
        hold   = phases.get("hold",   {}).get("frame")
        drop   = phases.get("drop",   {}).get("frame")

        cap = cv2.VideoCapture(str(mp4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        decoded = decode_frames(mp4, [pickup, hold, drop])
        strip   = make_strip(mp4.stem, pickup, hold, drop, fps, decoded)
        strips.append(strip)
        names.append(mp4.stem)

        print(f"  [{i+1}/{n}] {mp4.name}  pickup={pickup}  hold={hold}  drop={drop}")

    return strips, names


# ── viewer ────────────────────────────────────────────────────────────────────

def run_viewer(strips: list[np.ndarray]) -> None:
    n = len(strips)
    WIN = "Rollout Phase Viewer"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, STRIP_W, STRIP_H + 40)

    ep_pos = [0]

    def on_tb(val):
        ep_pos[0] = val

    cv2.createTrackbar("Rollout", WIN, 0, max(n - 1, 1), on_tb)

    while True:
        idx = ep_pos[0]
        cv2.imshow(WIN, strips[idx])

        key = cv2.waitKeyEx(20)
        if key == -1:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue

        if key in (2424832, 65361, ord("a")):    # LEFT
            ep_pos[0] = max(idx - 1, 0)
            cv2.setTrackbarPos("Rollout", WIN, ep_pos[0])
        elif key in (2555904, 65363, ord("d")):  # RIGHT
            ep_pos[0] = min(idx + 1, n - 1)
            cv2.setTrackbarPos("Rollout", WIN, ep_pos[0])
        elif key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=str(DEFAULT_DIR),
                        help="Directory containing rollout MP4s and phase JSONs")
    args = parser.parse_args()

    rollout_dir = Path(args.dir)
    if not rollout_dir.exists():
        print(f"Directory not found: {rollout_dir}")
        return

    print(f"Scanning {rollout_dir} ...")
    strips, names = build_strips(rollout_dir)
    if not strips:
        return

    print(f"\nLoaded {len(strips)} rollouts. Opening viewer...")
    run_viewer(strips)


if __name__ == "__main__":
    main()
