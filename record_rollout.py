"""Record a video from a USB camera and save it into ./rollouts/.

Usage:
    python record_rollout.py                  # camera 0, press 'q' to stop
    python record_rollout.py --camera 1       # pick a different USB cam
    python record_rollout.py --duration 15    # auto-stop after 15s

Requirements:
    pip install opencv-python
"""
import argparse
import os
import time
from datetime import datetime

import cv2


def record_video(
    camera_index: int = 0,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    duration: float | None = None,
    output_dir: str = "rollouts",
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {camera_index}. "
            "Check that the USB camera is connected and not in use by another app."
        )

    # Request the camera settings (the driver may silently override).
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"rollout_{timestamp}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, actual_fps, (actual_w, actual_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    print(f"Recording to {out_path} ({actual_w}x{actual_h} @ {actual_fps:.1f} fps)")
    print("Press 'q' in the preview window to stop."
          + (f" Auto-stop after {duration}s." if duration else ""))

    start = time.time()
    frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: failed to read frame, stopping.")
                break

            writer.write(frame)
            frames += 1

            cv2.imshow("Recording (press q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if duration is not None and (time.time() - start) >= duration:
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - start
    print(f"Saved {frames} frames ({elapsed:.1f}s) to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="USB camera index (default 0)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--duration", type=float, default=None,
                        help="Auto-stop after N seconds (default: run until 'q')")
    parser.add_argument("--output-dir", default="rollouts")
    args = parser.parse_args()

    record_video(
        camera_index=args.camera,
        fps=args.fps,
        width=args.width,
        height=args.height,
        duration=args.duration,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
