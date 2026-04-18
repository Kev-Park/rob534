"""
Orange object segmentation pipeline for video.
Outputs: binary mask frames, overlay video, bounding box CSV.

Usage:
    python segment_orange.py --input path/to/video.mp4 [--output_dir outputs/segmentation]
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np


# --- Orange HSV thresholds (OpenCV: H 0-180, S 0-255, V 0-255) ---
# Covers typical orange range. Tune ORANGE_LOWER/UPPER if needed.
ORANGE_LOWER1 = np.array([5,  100, 80])
ORANGE_UPPER1 = np.array([25, 255, 255])
# Second range to catch reddish-orange near hue=0/180
ORANGE_LOWER2 = np.array([0,  100, 80])
ORANGE_UPPER2 = np.array([5,  255, 255])

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def get_orange_mask(frame_hsv: np.ndarray) -> np.ndarray:
    """Return binary mask of orange pixels."""
    mask1 = cv2.inRange(frame_hsv, ORANGE_LOWER1, ORANGE_UPPER1)
    mask2 = cv2.inRange(frame_hsv, ORANGE_LOWER2, ORANGE_UPPER2)
    mask = cv2.bitwise_or(mask1, mask2)
    # Morphological cleanup: close small holes, remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL, iterations=1)
    return mask


def largest_contour_bbox(mask: np.ndarray):
    """Return (x, y, w, h) of largest contour, or None if nothing found."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 200:   # ignore tiny blobs
        return None, None
    return cv2.boundingRect(largest), largest


def draw_overlay(frame: np.ndarray, mask: np.ndarray, contour, bbox) -> np.ndarray:
    overlay = frame.copy()
    # Semi-transparent green fill over masked region
    color_layer = np.zeros_like(frame)
    color_layer[mask > 0] = (0, 200, 0)
    cv2.addWeighted(color_layer, 0.4, overlay, 1.0, 0, overlay)
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 140, 255), 2)
        cv2.putText(overlay, "orange", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    return overlay


def run(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    overlay_path = os.path.join(output_dir, "overlay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(overlay_path, fourcc, fps, (width, height))

    csv_path = os.path.join(output_dir, "bboxes.csv")
    json_path = os.path.join(output_dir, "bboxes.json")
    bbox_records = []

    frame_idx = 0
    print(f"Processing {total} frames from: {input_path}")
    print(f"Output dir: {output_dir}\n")

    with open(csv_path, "w", newline="") as csvfile:
        writer_csv = csv.writer(csvfile)
        writer_csv.writerow(["frame", "x", "y", "w", "h", "object_present"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = get_orange_mask(hsv)
            bbox, contour = largest_contour_bbox(mask)

            # Save binary mask
            mask_path = os.path.join(masks_dir, f"mask_{frame_idx:06d}.png")
            cv2.imwrite(mask_path, mask)

            # Draw and write overlay frame
            vis = draw_overlay(frame, mask, contour, bbox)
            writer.write(vis)

            # Record bbox
            if bbox is not None:
                x, y, w, h = bbox
                writer_csv.writerow([frame_idx, x, y, w, h, 1])
                bbox_records.append({"frame": frame_idx, "x": x, "y": y,
                                     "w": w, "h": h, "object_present": True})
            else:
                writer_csv.writerow([frame_idx, "", "", "", "", 0])
                bbox_records.append({"frame": frame_idx, "object_present": False})

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  {frame_idx}/{total} frames done...")

    cap.release()
    writer.release()

    with open(json_path, "w") as f:
        json.dump(bbox_records, f, indent=2)

    present = sum(1 for r in bbox_records if r["object_present"])
    print(f"\nDone. {frame_idx} frames processed, object detected in {present} frames.")
    print(f"  Masks  -> {masks_dir}/")
    print(f"  Overlay-> {overlay_path}")
    print(f"  CSV    -> {csv_path}")
    print(f"  JSON   -> {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orange object segmentation pipeline")
    parser.add_argument("--input",      required=True,  help="Path to input video file")
    parser.add_argument("--output_dir", default="outputs/segmentation",
                        help="Directory for all outputs (default: outputs/segmentation)")
    args = parser.parse_args()
    run(args.input, args.output_dir)
