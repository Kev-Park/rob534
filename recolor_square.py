"""
Recolor the segmented orange square in a video.

Pipeline per frame:
  1. Load binary mask, optionally dilate by --dilate pixels
  2. Blend overlay color at --alpha transparency directly over the original pixels
     (orange + green overlay → looks green; no grayscale step needed)

Usage:
    python recolor_square.py --input VIDEO --masks_dir MASKS_DIR --output OUT.avi
                             [--color R G B] [--alpha 0.5] [--dilate 1]
"""

import argparse
import os

import cv2
import numpy as np


def recolor_frame(frame: np.ndarray, mask: np.ndarray,
                  color_bgr: tuple, alpha: float, dilate: int = 1) -> np.ndarray:
    """
    Returns frame with the masked region recolored.
    Blends `color_bgr` at `alpha` directly over the original pixels —
    orange + green overlay at the right alpha reads as green.
    """
    out = frame.copy()

    if mask.max() == 0:
        return out

    # Dilate mask by `dilate` pixels for a slightly larger coverage
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate+1, 2*dilate+1))
        mask = cv2.dilate(mask, k, iterations=1)

    color_layer = np.full_like(frame, color_bgr, dtype=np.uint8)
    roi = out[mask > 0].astype(np.float32)
    col = color_layer[mask > 0].astype(np.float32)
    blended = (1.0 - alpha) * roi + alpha * col
    out[mask > 0] = np.clip(blended, 0, 255).astype(np.uint8)

    return out


def run(input_path: str, masks_dir: str, output_path: str,
        color_bgr: tuple, alpha: float, dilate: int = 1):

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")

    r, g, b = color_bgr  # stored as BGR internally
    print(f"Input  : {input_path}  ({total} frames)")
    print(f"Masks  : {masks_dir}")
    print(f"Output : {output_path}")
    print(f"Color  : RGB({b},{g},{r})  alpha={alpha}  dilate={dilate}px\n")   # note: stored BGR

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask_path = os.path.join(masks_dir, f"mask_{frame_idx:06d}.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)

        out = recolor_frame(frame, mask, (r, g, b), alpha, dilate)
        writer.write(out)

        frame_idx += 1
        if frame_idx % 1000 == 0:
            print(f"  {frame_idx}/{total}...")

    cap.release()
    writer.release()
    print(f"\nDone. {frame_idx} frames written to:\n  {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recolor segmented square in video")
    parser.add_argument("--input",      required=True,  help="Path to original video")
    parser.add_argument("--masks_dir",  required=True,  help="Folder of mask_XXXXXX.png files")
    parser.add_argument("--output",     required=True,  help="Output .avi path")
    parser.add_argument("--color", nargs=3, type=int, default=[0, 120, 255],
                        metavar=("R", "G", "B"),
                        help="Overlay color as R G B (default: 0 120 255 = blue)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Overlay transparency 0.0=invisible 1.0=solid (default: 0.5)")
    parser.add_argument("--dilate", type=int, default=1,
                        help="Dilate mask by this many pixels (default: 1)")
    args = parser.parse_args()

    r, g, b = args.color          # user gives RGB
    run(args.input, args.masks_dir, args.output,
        color_bgr=(b, g, r),      # OpenCV uses BGR
        alpha=args.alpha, dilate=args.dilate)
