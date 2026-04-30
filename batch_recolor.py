"""
Batch segmentation + recolor pipeline.

For each input video:
  1. Runs orange-square segmentation to produce per-frame masks
  2. Produces one recolored AVI per (color, alpha) combination

Output structure:
  <output_root>/
    <video_stem>/
      masks/mask_XXXXXX.png
      bboxes.csv / bboxes.json
      recolored/
        <color>_a<alpha>.avi
        ...

Usage:
    python batch_recolor.py --videos vid1.mp4 vid2.mp4 [--output_root outputs/batch]
                            [--alphas 0.3 0.5 0.7] [--dilate 1]
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from segment_orange import get_orange_mask, best_square_contour, make_contour_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[GPU] using {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

# ── color palette (name, R, G, B) ──────────────────────────────────────────
COLORS = [
    ("red",     220,  40,  40),
    ("orange",  255, 140,   0),
    ("yellow",  240, 220,  20),
    ("lime",     80, 220,  40),
    ("green",    30, 200,  30),
    ("teal",      0, 180, 160),
    ("cyan",     20, 210, 230),
    ("blue",      0, 100, 255),
    ("indigo",   60,  40, 220),
    ("purple",  160,  30, 200),
    ("pink",    240,  80, 160),
    ("white",   255, 255, 255),
    ("gray",    128, 128, 128),
]


# ── segmentation ────────────────────────────────────────────────────────────
def segment_video(input_path: str, out_dir: str):
    masks_dir = os.path.join(out_dir, "masks")
    csv_path  = os.path.join(out_dir, "bboxes.csv")

    # Skip if already done
    if os.path.isdir(masks_dir) and os.listdir(masks_dir):
        print(f"  [seg] masks already exist, skipping segmentation.")
        return masks_dir

    os.makedirs(masks_dir, exist_ok=True)
    cap   = cv2.VideoCapture(input_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  [seg] {total} frames -> {masks_dir}")

    bbox_records = []
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y", "width", "height", "area", "sq_score", "object_present"])
        idx = 0
        pbar = tqdm(total=total, desc="  segmenting", unit="fr", disable=False)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            orange_mask = get_orange_mask(hsv)
            bbox, contour, score = best_square_contour(orange_mask)
            sq_mask = make_contour_mask(frame.shape, contour)
            cv2.imwrite(os.path.join(masks_dir, f"mask_{idx:06d}.png"), sq_mask)
            if bbox:
                x, y, bw, bh = bbox
                w.writerow([idx, x, y, bw, bh, bw*bh, f"{score:.3f}", 1])
                bbox_records.append({"frame": idx, "x": x, "y": y, "w": bw, "h": bh,
                                     "area": bw*bh, "sq_score": round(score, 3), "object_present": True})
            else:
                w.writerow([idx, "", "", "", "", "", "", 0])
                bbox_records.append({"frame": idx, "object_present": False})
            idx += 1
            pbar.update(1)
        pbar.close()

    cap.release()
    with open(os.path.join(out_dir, "bboxes.json"), "w") as f:
        json.dump(bbox_records, f, indent=2)
    present = sum(1 for r in bbox_records if r["object_present"])
    print(f"  [seg] done. object in {present}/{idx} frames ({100*present/idx:.1f}%)")
    return masks_dir


# ── pack masks into a single video for fast sequential reading ───────────────
def build_mask_video(masks_dir: str, mask_video_path: str, total: int,
                     fps: float, W: int, H: int):
    """Save all mask PNGs as a single greyscale AVI for fast sequential reads."""
    if os.path.exists(mask_video_path) and os.path.getsize(mask_video_path) > 1_000_000:
        print(f"  [masks] mask video already exists, skipping.")
        return
    print(f"  [masks] packing {total} mask PNGs -> {mask_video_path}")
    writer = cv2.VideoWriter(mask_video_path, cv2.VideoWriter_fourcc(*"XVID"),
                             fps, (W, H), isColor=False)
    blank = np.zeros((H, W), dtype=np.uint8)
    for i in tqdm(range(total), desc="  packing masks", unit="fr", disable=False):
        mp = os.path.join(masks_dir, f"mask_{i:06d}.png")
        m  = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if os.path.exists(mp) else blank
        writer.write(m)
    writer.release()
    print(f"  [masks] done.")


# ── GPU batch recolor ────────────────────────────────────────────────────────
def gpu_blend_batch(frames_np: np.ndarray, masks_np: np.ndarray,
                    color_alphas: list, dilate: int) -> list:
    B, H, W, _ = frames_np.shape
    frames_t = torch.from_numpy(frames_np).to(DEVICE).float()
    masks_t  = torch.from_numpy(masks_np).to(DEVICE).float() / 255.0

    if dilate > 0:
        m = masks_t.unsqueeze(1)
        m = F.max_pool2d(m, kernel_size=2*dilate+1, stride=1, padding=dilate)
        masks_t = m.squeeze(1)

    mask_3d = masks_t.unsqueeze(-1)  # (B,H,W,1)

    results = []
    for color_bgr, alpha in color_alphas:
        color = torch.tensor(list(color_bgr), dtype=torch.float32, device=DEVICE)
        blended = (1.0 - alpha) * frames_t + alpha * color.view(1, 1, 1, 3)
        out = torch.where(mask_3d > 0.5, blended, frames_t)
        results.append(out.clamp(0, 255).byte().cpu().numpy())
    return results


def recolor_all(input_path: str, masks_dir: str, recolor_dir: str,
                alphas: list, dilate: int,
                batch_size: int = 32, group_size: int = 6):
    os.makedirs(recolor_dir, exist_ok=True)

    jobs = []
    for label, R, G, B in COLORS:
        for alpha in alphas:
            fname = f"{label}_a{alpha:.1f}.avi"
            path  = os.path.join(recolor_dir, fname)
            if os.path.exists(path) and os.path.getsize(path) > 10_000_000:
                print(f"  [recolor] skip: {fname}")
                continue
            jobs.append((label, B, G, R, alpha, path))

    if not jobs:
        print("  [recolor] all files already exist, skipping.")
        return

    cap   = cv2.VideoCapture(input_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Build mask video once — sequential read replaces 38k PNG file reads
    mask_video = os.path.join(masks_dir, "_masks.avi")
    build_mask_video(masks_dir, mask_video, total, fps, W, H)

    fourcc   = cv2.VideoWriter_fourcc(*"XVID")
    n_groups = (len(jobs) + group_size - 1) // group_size
    print(f"  [recolor] {len(jobs)} outputs | {n_groups} passes | "
          f"batch={batch_size} group={group_size} device={DEVICE}")

    for g_idx, g_start in enumerate(range(0, len(jobs), group_size), 1):
        group     = jobs[g_start : g_start + group_size]
        labels    = [j[0] for j in group]
        ca_pairs  = [((j[1], j[2], j[3]), j[4]) for j in group]
        out_paths = [j[5] for j in group]

        print(f"  [pass {g_idx}/{n_groups}] {', '.join(labels)}")
        writers  = [cv2.VideoWriter(p, fourcc, fps, (W, H)) for p in out_paths]
        cap_src  = cv2.VideoCapture(input_path)
        cap_mask = cv2.VideoCapture(mask_video)

        frame_buf, mask_buf, idx = [], [], 0
        pbar = tqdm(total=total, desc=f"  pass {g_idx}/{n_groups}", unit="fr", disable=False)

        def flush(fb, mb):
            outs = gpu_blend_batch(np.stack(fb), np.stack(mb), ca_pairs, dilate)
            for w, out_batch in zip(writers, outs):
                for f in out_batch:
                    w.write(f)

        while True:
            ret_f, frame = cap_src.read()
            ret_m, mask  = cap_mask.read()
            if not ret_f:
                if frame_buf:
                    flush(frame_buf, mask_buf)
                break
            if not ret_m:
                mask = np.zeros((H, W, 3), dtype=np.uint8)
            # mask from video may be 3-channel due to codec; take first channel
            m = mask[:, :, 0] if mask.ndim == 3 else mask
            frame_buf.append(frame)
            mask_buf.append(m)
            idx += 1
            pbar.update(1)
            if len(frame_buf) >= batch_size:
                flush(frame_buf, mask_buf)
                frame_buf, mask_buf = [], []
        pbar.close()

        cap_src.release()
        cap_mask.release()
        for w in writers:
            w.release()

    print(f"  [recolor] done -> {recolor_dir}/")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Batch segment + recolor pipeline")
    parser.add_argument("--videos",      nargs="+", required=True, help="Input video paths")
    parser.add_argument("--output_root", default="outputs/batch",  help="Root output directory")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--dilate",      type=int,   default=1)
    parser.add_argument("--batch_size",  type=int,   default=32,
                        help="Frames per GPU batch (default 32)")
    parser.add_argument("--group_size",  type=int,   default=6,
                        help="Writers open simultaneously (default 6)")
    args = parser.parse_args()

    print(f"Videos : {args.videos}")
    print(f"Alphas : {args.alphas}")
    print(f"Colors : {len(COLORS)}")
    print(f"Output : {args.output_root}\n")

    for video_path in args.videos:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        vid_dir     = os.path.join(args.output_root, stem)
        recolor_dir = os.path.join(vid_dir, "recolored")
        print(f"=== {stem} ===")

        masks_dir = segment_video(video_path, vid_dir)
        recolor_all(video_path, masks_dir, recolor_dir, args.alphas, args.dilate,
                    batch_size=args.batch_size, group_size=args.group_size)
        print()

    print("All done.")


if __name__ == "__main__":
    main()
