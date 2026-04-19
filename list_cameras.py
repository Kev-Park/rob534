"""Probe camera indices 0..N-1 and print which ones deliver frames.

Usage:
    python list_cameras.py          # probe 0..5
    python list_cameras.py --max 10 # probe 0..9
    python list_cameras.py --show 1 # open preview window on the chosen index

Tip (macOS): `system_profiler SPCameraDataType` lists attached cameras by name,
but it does not tell you the OpenCV index. Indices depend on enumeration order,
so this script is the reliable way to map name -> index for cv2.VideoCapture.
"""
import argparse

import cv2


def probe(max_index: int) -> list[int]:
    working = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  index {i}: OK  ({w}x{h} @ {fps:.1f} fps)")
            working.append(i)
        else:
            print(f"  index {i}: opened but no frame")
        cap.release()
    return working


def preview(index: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}")
    print(f"Previewing camera {index}. Press 'q' to close.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(f"camera {index} (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=6, help="Probe indices 0..MAX-1")
    parser.add_argument("--show", type=int, default=None, help="Preview this index after probing")
    args = parser.parse_args()

    print(f"Probing camera indices 0..{args.max - 1}...")
    working = probe(args.max)
    print(f"\nWorking indices: {working}" if working else "\nNo working cameras found.")

    if args.show is not None:
        preview(args.show)


if __name__ == "__main__":
    main()
