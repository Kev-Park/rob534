"""
Orange object segmentation pipeline for video.
Selects the orange contour that best matches a square shape.
Detects pick-up / drop events by looking for black clamp pixels adjacent to the object.
Outputs: binary mask frames, overlay video, bounding box CSV, events CSV/JSON,
         depth-over-time plot, and composite video with rolling depth panel.

Usage:
    python segment_orange.py --input path/to/video.mp4 [--output_dir outputs/segmentation] [--area 8000]

--area N          Expected area of the square in pixels² (optional).
--real_size_mm N  Real edge length of the orange square in mm. When provided,
                  depth is reported in mm using the pinhole model. Without it,
                  a unitless apparent-size proxy is used (larger = closer).
--focal_px N      Camera focal length in pixels (default: 500, typical for 640px-wide webcam).
                  Only used when --real_size_mm is set.
--clamp_expand    Pixels to look outside the bbox for dark clamp pixels (default: 15)
--dark_thresh     HSV Value threshold below which a pixel is "black/dark" (default: 50)
--dark_ratio      Fraction of border pixels that must be dark to count as held (default: 0.25)
--confirm_frames  Consecutive frames required to confirm a state change (default: 3)
"""

import argparse
import csv
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def squareness_score(contour, target_area: float = None) -> float:
    """
    Score a contour on how square-like it is (0–1, higher = more square).
    Combines:
      - aspect_ratio : bounding box min/max side ratio  (1.0 = perfect square)
      - extent       : contour area / bbox area         (1.0 = fully filled)
      - solidity     : contour area / convex hull area  (1.0 = fully convex)
      - area_score   : proximity to target_area         (1.0 = exact match)
    """
    area = cv2.contourArea(contour)
    if area < 200:
        return 0.0
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = min(w, h) / max(w, h)          # 1.0 = square bbox
    extent       = area / (w * h)                  # 1.0 = fills bbox
    hull         = cv2.convexHull(contour)
    hull_area    = cv2.contourArea(hull)
    solidity     = area / hull_area if hull_area > 0 else 0.0

    score = aspect_ratio * extent * solidity

    if target_area is not None:
        # Gaussian-like penalty: score drops as area deviates from target
        area_score = np.exp(-0.5 * ((area - target_area) / (target_area * 0.5)) ** 2)
        score *= area_score

    return score


def best_square_contour(mask: np.ndarray, target_area: float = None):
    """Return (bbox, contour, score) of the most square-like orange blob."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0.0
    scored = [(c, squareness_score(c, target_area)) for c in contours]
    best_contour, best_score = max(scored, key=lambda x: x[1])
    if best_score < 0.05:   # nothing remotely square found
        return None, None, 0.0
    return cv2.boundingRect(best_contour), best_contour, best_score


def detect_held(frame_hsv: np.ndarray, bbox,
                expand: int = 15,
                dark_thresh: int = 50,
                min_dark_ratio: float = 0.25) -> bool:
    """
    Return True if the orange object appears to be gripped by black clamps.

    Looks at a border strip of width `expand` pixels just outside the bounding
    box for dark pixels (Value channel < dark_thresh).  If the fraction of dark
    pixels in that strip exceeds min_dark_ratio the object is considered held.
    """
    if bbox is None:
        return False

    x, y, w, h = bbox
    H, W = frame_hsv.shape[:2]

    # Outer region (bbox expanded by `expand` on all sides)
    ox1 = max(0, x - expand)
    oy1 = max(0, y - expand)
    ox2 = min(W, x + w + expand)
    oy2 = min(H, y + h + expand)

    # Value channel of expanded region
    roi_v = frame_hsv[oy1:oy2, ox1:ox2, 2].copy()

    # Blank out the inner bbox so we only examine the border strip
    ix1 = x - ox1
    iy1 = y - oy1
    ix2 = ix1 + w
    iy2 = iy1 + h
    roi_v[iy1:iy2, ix1:ix2] = 255  # set inner to bright so it won't count

    border_pixels = roi_v.ravel()
    dark_count = int(np.sum(border_pixels < dark_thresh))
    ratio = dark_count / len(border_pixels) if len(border_pixels) > 0 else 0.0
    return ratio >= min_dark_ratio


def draw_overlay(frame: np.ndarray, square_mask: np.ndarray, contour, bbox,
                 score: float, held: bool, holding_stable: bool) -> np.ndarray:
    """
    Three states:
      free           (held=False)               -> green fill, blue box
      gripped/moving (held=True, stable=False)  -> cyan fill, cyan box
      holding stable (held=True, stable=True)   -> yellow fill, yellow box
    """
    overlay = frame.copy()
    color_layer = np.zeros_like(frame)
    if holding_stable:
        fill_color = (0, 220, 220)    # yellow (BGR)
        box_color  = (0, 200, 200)
        state_tag  = " HOLDING"
    elif held:
        fill_color = (200, 200, 0)    # cyan
        box_color  = (200, 200, 0)
        state_tag  = " GRIPPED"
    else:
        fill_color = (0, 200, 0)      # green
        box_color  = (0, 140, 255)
        state_tag  = ""
    color_layer[square_mask > 0] = fill_color
    cv2.addWeighted(color_layer, 0.4, overlay, 1.0, 0, overlay)
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, fill_color, 2)
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, 2)
        label = f"sq={score:.2f} a={w*h}{state_tag}"
        cv2.putText(overlay, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    return overlay


def make_contour_mask(shape, contour) -> np.ndarray:
    """Binary mask containing only the selected contour filled in."""
    m = np.zeros(shape[:2], dtype=np.uint8)
    if contour is not None:
        cv2.drawContours(m, [contour], -1, 255, thickness=cv2.FILLED)
    return m


def estimate_depth(bbox, focal_px: float, real_size_mm: float | None):
    """
    Estimate distance from camera to object centre using the pinhole model:
        Z = f * W_real / W_apparent
    apparent size = average of bbox width and height (robust to mild aspect distortion).
    Returns depth in mm if real_size_mm is given, else a unitless proxy (larger = closer).
    """
    if bbox is None:
        return None
    _, _, w, h = bbox
    apparent = (w + h) / 2.0
    if apparent <= 0:
        return None
    if real_size_mm is not None:
        return focal_px * real_size_mm / apparent   # mm
    else:
        return focal_px / apparent                  # unitless proxy


def render_composite_video(overlay_path: str, composite_path: str,
                           depth_history: list, fps: float, events: list,
                           holding_history: list, video_h: int, unit_label: str):
    """
    Second pass: read overlay video and write composite with a static depth plot
    on the right where a red dot moves along the curve in sync with the video.
    """
    # --- Build full matplotlib plot ---
    dpi = 96
    fig_h_in = video_h / dpi
    fig_w_in = fig_h_in * 1.5
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#1a1a1a")

    times  = [i / fps for i in range(len(depth_history))]
    valid_t = [t for t, d in zip(times, depth_history) if d is not None]
    valid_d = [d for d in depth_history if d is not None]

    if valid_t:
        ax.plot(valid_t, valid_d, color="#00d480", linewidth=1.0, zorder=2)

    _shade_holding_regions(ax, holding_history, fps)

    for ev in events:
        color = "#50dd50" if ev["event"] == "pickup" else "#5050ee"
        ax.axvline(ev["time_s"], color=color, linewidth=0.9, alpha=0.8, zorder=1)
        ax.text(ev["time_s"] + 0.2, ax.get_ylim()[1] if valid_d else 0,
                ev["event"][0].upper(), color=color, fontsize=6, va="top")

    ax.set_xlabel("time (s)", color="#aaaaaa", fontsize=8)
    ax.set_ylabel(unit_label, color="#aaaaaa", fontsize=7)
    ax.set_title("Object depth over time", color="#dddddd", fontsize=9)
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    plt.tight_layout(pad=0.6)

    # Capture axis limits AFTER tight_layout (they won't change)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Render to numpy array
    fig.canvas.draw()
    fig_w_px, fig_h_px = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    base_plot = buf.reshape(fig_h_px, fig_w_px, 4)
    base_plot = cv2.cvtColor(base_plot, cv2.COLOR_RGBA2BGR)

    # Get axis bbox in pixel coordinates
    bbox   = ax.get_position()          # (x0, y0, width, height) in figure fractions
    ax_l   = bbox.x0 * fig_w_px
    ax_b   = bbox.y0 * fig_h_px        # from figure bottom
    ax_w   = bbox.width  * fig_w_px
    ax_h   = bbox.height * fig_h_px
    plt.close(fig)

    def data_to_px(t, d):
        """Map (time, depth) data coords → image (x, y) pixel coords."""
        x = ax_l + (t - x_min) / (x_max - x_min) * ax_w
        # image y=0 is top; matplotlib y=0 is bottom
        y = (fig_h_px - ax_b) - (d - y_min) / (y_max - y_min) * ax_h
        return (int(np.clip(x, 0, fig_w_px - 1)),
                int(np.clip(y, 0, fig_h_px - 1)))

    # Pre-compute red dot positions for every frame
    dot_pos = []
    for i, d in enumerate(depth_history):
        if d is not None:
            dot_pos.append(data_to_px(times[i], d))
        else:
            dot_pos.append(None)

    # Resize base_plot to exactly match video height
    if fig_h_px != video_h:
        scale = video_h / fig_h_px
        base_plot = cv2.resize(base_plot,
                               (int(fig_w_px * scale), video_h),
                               interpolation=cv2.INTER_AREA)
        fig_w_px, fig_h_px = base_plot.shape[1], base_plot.shape[0]
        # Scale dot positions too
        dot_pos = [
            (int(p[0] * scale), int(p[1] * scale)) if p else None
            for p in dot_pos
        ]

    panel_w = fig_w_px

    # --- Second pass: read overlay + write composite ---
    cap = cv2.VideoCapture(overlay_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    comp_w = vid_w + panel_w
    writer = cv2.VideoWriter(composite_path, fourcc, fps, (comp_w, video_h))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Rendering composite video ({total} frames)...")
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        panel = base_plot.copy()
        pos = dot_pos[fi] if fi < len(dot_pos) else None
        if pos is not None:
            cv2.circle(panel, pos, 7, (0, 0, 255), -1)       # red dot
            cv2.circle(panel, pos, 8, (255, 255, 255), 1)     # white ring

        writer.write(np.concatenate([frame, panel], axis=1))
        fi += 1
        if fi % 500 == 0:
            print(f"  {fi}/{total}...")

    cap.release()
    writer.release()
    print(f"  Composite -> {composite_path}")


def _shade_holding_regions(ax, holding_history: list, fps: float):
    """Add yellow shaded bands where holding_stable is True."""
    in_band = False
    start_t = 0.0
    for i, h in enumerate(holding_history):
        t = i / fps
        if h and not in_band:
            start_t = t
            in_band = True
        elif not h and in_band:
            ax.axvspan(start_t, t, color="#ddcc00", alpha=0.15, zorder=0)
            in_band = False
    if in_band:
        ax.axvspan(start_t, len(holding_history) / fps,
                   color="#ddcc00", alpha=0.15, zorder=0)


def save_depth_plot(depth_history: list, fps: float, events: list,
                    holding_history: list, out_path: str, unit_label: str):
    """Save a full-resolution matplotlib depth-over-time plot with event markers."""
    times  = [i / fps for i in range(len(depth_history))]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#1a1a1a")

    valid_t = [t for t, d in zip(times, depth_history) if d is not None]
    valid_d = [d for d in depth_history if d is not None]
    if valid_t:
        ax.plot(valid_t, valid_d, color="#00d480", linewidth=1.2)

    _shade_holding_regions(ax, holding_history, fps)

    for ev in events:
        color = "#50dd50" if ev["event"] == "pickup" else "#5050dd"
        ax.axvline(ev["time_s"], color=color, linewidth=1.2, alpha=0.8)
        ax.text(ev["time_s"], ax.get_ylim()[1] if valid_d else 0,
                ev["event"][0].upper(), color=color, fontsize=7, va="top")

    ax.set_xlabel("time (s)", color="#aaaaaa")
    ax.set_ylabel(unit_label, color="#aaaaaa")
    ax.set_title("Object depth over time  [yellow = holding stable]", color="#dddddd")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Depth plot -> {out_path}")


def run(input_path: str, output_dir: str,
        target_area: float = None,
        clamp_expand: int = 15,
        dark_thresh: int = 50,
        dark_ratio: float = 0.25,
        confirm_frames: int = 3,
        real_size_mm: float = None,
        focal_px: float = 500.0,
        stable_frames: int = 5):

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

    overlay_path    = os.path.join(output_dir, "overlay.mp4")
    composite_path  = os.path.join(output_dir, "depth_composite.mp4")
    depth_plot_path = os.path.join(output_dir, "depth_plot.png")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(overlay_path, fourcc, fps, (width, height))

    unit_label      = "depth (mm)" if real_size_mm else "depth (proxy, larger=closer)"
    depth_history   = []   # one entry per frame, None when object not visible
    holding_history = []   # one bool per frame: True = holding stable

    csv_path   = os.path.join(output_dir, "bboxes.csv")
    json_path  = os.path.join(output_dir, "bboxes.json")
    ev_csv_path  = os.path.join(output_dir, "events.csv")
    ev_json_path = os.path.join(output_dir, "events.json")

    bbox_records = []
    events       = []

    # --- Hysteresis state machine: grip ---
    confirmed_held    = False
    candidate_state   = False
    candidate_streak  = 0

    # --- Holding-stable state machine ---
    # When the arm holds the object close to the camera the orange detection
    # fails (clamp occludes it / too close).  So: bbox absent for >= stable_frames
    # consecutive frames = object is being held.  Reappearance ends the state.
    holding_stable = False
    absent_streak  = 0

    frame_idx = 0
    print(f"Processing {total} frames from: {input_path}")
    print(f"Output dir: {output_dir}")
    if target_area:
        print(f"Target area: {target_area:.0f} px²")
    print(f"Clamp params: expand={clamp_expand}px  dark_thresh={dark_thresh}  "
          f"dark_ratio={dark_ratio}  confirm_frames={confirm_frames}")
    print()

    with open(csv_path, "w", newline="") as csvfile, \
         open(ev_csv_path, "w", newline="") as ev_csvfile:

        writer_csv = csv.writer(csvfile)
        writer_csv.writerow(["frame", "time_s", "x", "y", "w", "h",
                              "area", "sq_score", "object_present", "held",
                              "holding_stable", "depth"])

        ev_writer = csv.writer(ev_csvfile)
        ev_writer.writerow(["frame", "time_s", "event"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            time_s = frame_idx / fps

            hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            orange_mask = get_orange_mask(hsv)
            bbox, contour, score = best_square_contour(orange_mask, target_area)

            # Raw held signal for this frame
            raw_held = detect_held(hsv, bbox,
                                   expand=clamp_expand,
                                   dark_thresh=dark_thresh,
                                   min_dark_ratio=dark_ratio)

            # --- Hysteresis: require confirm_frames consecutive frames ---
            if raw_held == candidate_state:
                candidate_streak += 1
            else:
                candidate_state  = raw_held
                candidate_streak = 1

            if candidate_streak >= confirm_frames and candidate_state != confirmed_held:
                confirmed_held = candidate_state
                event_name = "pickup" if confirmed_held else "drop"
                ev_writer.writerow([frame_idx, f"{time_s:.3f}", event_name])
                events.append({"frame": frame_idx, "time_s": round(time_s, 3),
                                "event": event_name})
                print(f"  [{event_name.upper():6s}] frame {frame_idx:5d}  t={time_s:.2f}s")

            square_mask = make_contour_mask(frame.shape, contour)

            # --- Holding-stable detection ---
            # Object absent (bbox is None) for >= stable_frames frames = being held.
            # Object reappearing = released.
            if bbox is None:
                absent_streak += 1
                if absent_streak >= stable_frames:
                    holding_stable = True
            else:
                absent_streak  = 0
                holding_stable = False

            # Depth estimate
            depth = estimate_depth(bbox, focal_px, real_size_mm)
            depth_history.append(depth)
            holding_history.append(holding_stable)

            # Masks, overlay, CSV
            mask_path = os.path.join(masks_dir, f"mask_{frame_idx:06d}.png")
            cv2.imwrite(mask_path, square_mask)

            vis = draw_overlay(frame, square_mask, contour, bbox, score,
                               confirmed_held, holding_stable)

            # Depth text on overlay
            if depth is not None:
                d_label = f"depth: {depth:.0f} {'mm' if real_size_mm else 'px-1'}"
                cv2.putText(vis, d_label, (8, height - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2)

            writer.write(vis)

            if bbox is not None:
                x, y, w, h = bbox
                writer_csv.writerow([frame_idx, f"{time_s:.3f}",
                                     x, y, w, h, w * h, f"{score:.3f}", 1,
                                     int(confirmed_held), int(holding_stable),
                                     f"{depth:.2f}" if depth is not None else ""])
                bbox_records.append({"frame": frame_idx, "time_s": round(time_s, 3),
                                     "x": x, "y": y, "w": w, "h": h, "area": w * h,
                                     "sq_score": round(score, 3),
                                     "object_present": True,
                                     "held": confirmed_held,
                                     "holding_stable": holding_stable,
                                     "depth": round(depth, 2) if depth is not None else None})
            else:
                writer_csv.writerow([frame_idx, f"{time_s:.3f}",
                                     "", "", "", "", "", "", 0,
                                     int(confirmed_held), int(holding_stable), ""])
                bbox_records.append({"frame": frame_idx, "time_s": round(time_s, 3),
                                     "object_present": False, "held": confirmed_held,
                                     "holding_stable": holding_stable, "depth": None})

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  {frame_idx}/{total} frames done...")

    cap.release()
    writer.release()

    with open(json_path, "w") as f:
        json.dump(bbox_records, f, indent=2)

    with open(ev_json_path, "w") as f:
        json.dump(events, f, indent=2)

    save_depth_plot(depth_history, fps, events, holding_history, depth_plot_path, unit_label)
    render_composite_video(overlay_path, composite_path,
                           depth_history, fps, events, holding_history,
                           video_h=height, unit_label=unit_label)

    present = sum(1 for r in bbox_records if r["object_present"])
    print(f"\nDone. {frame_idx} frames processed, object detected in {present} frames.")
    print(f"  Masks       -> {masks_dir}/")
    print(f"  Overlay     -> {overlay_path}")
    print(f"  Composite   -> {composite_path}")
    print(f"  Depth plot  -> {depth_plot_path}")
    print(f"  CSV         -> {csv_path}")
    print(f"  JSON        -> {json_path}")
    print(f"  Events      -> {ev_csv_path}  /  {ev_json_path}")
    print(f"\n  {len(events)} grip event(s) detected:")
    for ev in events:
        print(f"    [{ev['event'].upper():6s}] frame {ev['frame']:5d}  t={ev['time_s']:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orange object segmentation pipeline")
    parser.add_argument("--input",       required=True,  help="Path to input video file")
    parser.add_argument("--output_dir",  default="outputs/segmentation",
                        help="Directory for all outputs (default: outputs/segmentation)")
    parser.add_argument("--area",        type=float, default=None,
                        help="Expected area of the square in pixels² (optional)")
    parser.add_argument("--clamp_expand", type=int, default=15,
                        help="Pixels to expand bbox when searching for dark clamp pixels (default: 15)")
    parser.add_argument("--dark_thresh", type=int, default=50,
                        help="HSV Value threshold for 'dark/black' pixels (default: 50)")
    parser.add_argument("--dark_ratio",  type=float, default=0.25,
                        help="Min fraction of border pixels that must be dark to count as held (default: 0.25)")
    parser.add_argument("--confirm_frames", type=int, default=3,
                        help="Consecutive frames required to confirm pick-up/drop (default: 3)")
    parser.add_argument("--real_size_mm", type=float, default=None,
                        help="Real edge length of the orange square in mm. "
                             "Enables metric depth output. Without it a unitless proxy is used.")
    parser.add_argument("--focal_px", type=float, default=500.0,
                        help="Camera focal length in pixels (default: 500). "
                             "Only used when --real_size_mm is set.")
    parser.add_argument("--stable_frames", type=int, default=5,
                        help="Consecutive stable frames required to confirm holding state (default: 5)")
    args = parser.parse_args()
    run(args.input, args.output_dir,
        target_area=args.area,
        clamp_expand=args.clamp_expand,
        dark_thresh=args.dark_thresh,
        dark_ratio=args.dark_ratio,
        confirm_frames=args.confirm_frames,
        real_size_mm=args.real_size_mm,
        focal_px=args.focal_px,
        stable_frames=args.stable_frames)
