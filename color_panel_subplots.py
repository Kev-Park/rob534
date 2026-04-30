"""
Recreate the Color x Alpha panel visualization.
Shows frame 11540 from file-001 recolored with all 13 colors × 4 alpha values.

Output: color_panel_subplots.png  (saved next to this script)
"""

import os
import sys

import av
import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────────
ORIG_VIDEO = (
    r"C:\Users\calle\.cache\huggingface\hub"
    r"\datasets--nc8304--so101_combined_cubeONLY"
    r"\snapshots\acc242c231f60171a5b2833442d176cd793ea8c9"
    r"\videos\observation.images.front\chunk-000\file-001.mp4"
)
MASK_PATH = (
    r"C:\Users\calle\PycharmProjects\RobotArm"
    r"\outputs\batch\file-001\masks\mask_011540.png"
)
FRAME_IDX = 11540

# ── color palette (name, R, G, B) ────────────────────────────────────────────
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

COLORS = [
    # ("red",     220,  40,  40),
    # ("orange",  255, 140,   0),
    ("yellow",  240, 220,  20),
    # ("lime",     80, 220,  40),
    ("green",    30, 200,  30),
    ("teal",      0, 180, 160),
    # ("cyan",     20, 210, 230),
    # ("blue",      0, 100, 255),
    ("indigo",   60,  40, 220),
    ("purple",  160,  30, 200),
    ("pink",    240,  80, 160),
#     ("white",   255, 255, 255),
#     ("gray",    128, 128, 128),
]

ALPHAS = [0.3, 0.5, 0.7, 0.9]


# ── helpers ──────────────────────────────────────────────────────────────────
def recolor_frame(frame: np.ndarray, mask: np.ndarray,
                  color_bgr: tuple, alpha: float, dilate: int = 1) -> np.ndarray:
    out = frame.copy()
    if mask.max() == 0:
        return out
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        mask = cv2.dilate(mask, k, iterations=1)
    color_layer = np.full_like(frame, color_bgr, dtype=np.uint8)
    roi = out[mask > 0].astype(np.float32)
    col = color_layer[mask > 0].astype(np.float32)
    blended = (1.0 - alpha) * roi + alpha * col
    out[mask > 0] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def extract_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Seek to frame_idx using PyAV (works with AV1 on Windows)."""
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONREF"  # faster seeking
        fps = float(stream.average_rate or 30)
        # Seek to a timestamp slightly before the target
        target_sec = max(0.0, (frame_idx - 1) / fps)
        target_ts = int(target_sec / stream.time_base)
        container.seek(target_ts, stream=stream)

        for i, frame in enumerate(container.decode(stream)):
            pts_frame = int(frame.pts * stream.time_base * fps) if frame.pts is not None else -1
            if pts_frame >= frame_idx or i >= 5:
                return frame.to_ndarray(format="bgr24")

    raise ValueError(f"Could not extract frame {frame_idx} from {video_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Extracting frame {FRAME_IDX} from {os.path.basename(ORIG_VIDEO)} ...")
    frame_bgr = extract_frame(ORIG_VIDEO, FRAME_IDX)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    print(f"Frame shape: {frame_bgr.shape}")

    mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        sys.exit(f"Could not load mask: {MASK_PATH}")
    print(f"Mask loaded: {mask.shape}, non-zero pixels: {np.count_nonzero(mask)}")

    n_rows = 1 + len(COLORS)   # original + 13 colors
    n_cols = len(ALPHAS)        # 4 alphas

    LABEL_FS = 16   # color row-label font size
    ALPHA_FS = 13   # alpha-below-image font size
    TITLE_FS = 18   # main title font size

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.8, n_rows * 2.4))
    fig.patch.set_facecolor("#111111")
    fig.suptitle(f"Color x Alpha panel  (frame {FRAME_IDX})",
                 color="white", fontsize=TITLE_FS, y=1.002)

    row_labels = ["original"] + [name for name, *_ in COLORS]

    for i in range(n_rows):
        is_original = (i == 0)
        name = row_labels[i]

        for j, alpha in enumerate(ALPHAS):
            ax = axes[i, j]

            if is_original:
                ax.imshow(frame_rgb)
            else:
                _, R, G, B = COLORS[i - 1]
                recolored = recolor_frame(frame_bgr, mask.copy(), (B, G, R), alpha)
                ax.imshow(cv2.cvtColor(recolored, cv2.COLOR_BGR2RGB))

            ax.axis("off")

            # alpha value below each image
            ax.text(0.5, -0.06, f"α = {alpha}",
                    transform=ax.transAxes,
                    ha="center", va="top",
                    color="white", fontsize=ALPHA_FS)

            # color label on the left side of the first column only
            if j == 0:
                ax.text(-0.08, 0.5, name,
                        transform=ax.transAxes,
                        ha="right", va="center",
                        color="white", fontsize=LABEL_FS, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.subplots_adjust(left=0.10, hspace=0.35, wspace=0.05)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "color_panel_subplots.png")
    plt.savefig(out_path, dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
