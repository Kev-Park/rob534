import cv2
import torch
import numpy as np
import os
from pathlib import Path
from PIL import Image
from transformers import Sam3Processor, Sam3Model

# --- Configuration ---
TEXT_PROMPT = "the block actively being grasped"
VIDEO_INPUT = "file-000-h264.mp4"
VIDEO_OUTPUT = "file-000-segmented.mp4"
# Score threshold filters predicted instances before mask extraction.
SCORE_THRESHOLD = float(os.environ.get("SAM3_SCORE_THRESHOLD", "0.05"))
# Mask threshold binarizes per-pixel mask logits.
MASK_THRESHOLD = float(os.environ.get("SAM3_MASK_THRESHOLD", "0.5"))
PRINT_DEBUG = os.environ.get("SAM3_DEBUG", "1") == "1"
TRACK_SINGLE_INSTANCE = os.environ.get("SAM3_SINGLE_INSTANCE", "1") == "1"
TRACK_IOU_WEIGHT = float(os.environ.get("SAM3_TRACK_IOU_WEIGHT", "0.5"))
MODEL_REF = os.environ.get("SAM3_MODEL_REF", "facebook/sam3")
MODEL_PATH = os.environ.get("SAM3_MODEL_PATH")
HF_CACHE_DIR = os.environ.get("HF_HOME")

# Default to local-only loading on cluster jobs, or when offline env vars are set.
SAM3_LOCAL_ONLY = os.environ.get("SAM3_LOCAL_ONLY")
if SAM3_LOCAL_ONLY is None:
    SAM3_LOCAL_ONLY = (
        os.environ.get("HF_HUB_OFFLINE") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        or os.environ.get("SLURM_JOB_ID") is not None
    )
else:
    SAM3_LOCAL_ONLY = SAM3_LOCAL_ONLY == "1"


def resolve_local_sam3_source() -> str:
    """Resolve a local SAM3 snapshot path when the repo was downloaded ahead of time."""
    if MODEL_PATH:
        candidate = Path(MODEL_PATH).expanduser()
        if candidate.is_dir():
            return str(candidate)

    candidate = Path(MODEL_REF).expanduser()
    if candidate.is_dir():
        return str(candidate)

    if HF_CACHE_DIR:
        snapshots_dir = Path(HF_CACHE_DIR).expanduser() / "hub" / "models--facebook--sam3" / "snapshots"
        if snapshots_dir.is_dir():
            snapshots = sorted(
                (path for path in snapshots_dir.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for snapshot_dir in snapshots:
                if (snapshot_dir / "config.json").is_file():
                    return str(snapshot_dir)

    return MODEL_REF


MODEL_SOURCE = resolve_local_sam3_source()


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def resolve_video_path(path_value: str, *, must_exist: bool) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        resolved = candidate
    else:
        resolved = (Path.cwd() / candidate)

    resolved = resolved.resolve()
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"Video path does not exist: {resolved}")
    return resolved


VIDEO_INPUT_PATH = resolve_video_path(os.environ.get("SAM3_VIDEO_INPUT", VIDEO_INPUT), must_exist=True)
VIDEO_OUTPUT_PATH = resolve_video_path(os.environ.get("SAM3_VIDEO_OUTPUT", VIDEO_OUTPUT), must_exist=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Working directory: {Path.cwd()}")
print(f"Input video: {VIDEO_INPUT_PATH}")
print(f"Output video: {VIDEO_OUTPUT_PATH}")
print(
    f"Prompt='{TEXT_PROMPT}', score_threshold={SCORE_THRESHOLD}, "
    f"mask_threshold={MASK_THRESHOLD}, single_instance={TRACK_SINGLE_INSTANCE}"
)

# 1. Load SAM 3 from Hugging Face
# SAM 3 natively understands text, so no Grounding DINO is needed.
print(
    f"Loading SAM3 from '{MODEL_SOURCE}' "
    f"(local_only={SAM3_LOCAL_ONLY}, cache_dir={HF_CACHE_DIR})"
)
try:
    model = Sam3Model.from_pretrained(
        MODEL_SOURCE,
        cache_dir=HF_CACHE_DIR,
        local_files_only=SAM3_LOCAL_ONLY,
    ).to(device)
    processor = Sam3Processor.from_pretrained(
        MODEL_SOURCE,
        cache_dir=HF_CACHE_DIR,
        local_files_only=SAM3_LOCAL_ONLY,
    )
except OSError as exc:
    raise RuntimeError(
        "Failed to load SAM3 locally. Make sure the full repository snapshot exists under "
        "HF_HOME/hub/models--facebook--sam3/snapshots/<sha>/ and contains config.json plus processor files. "
        "If you need to refresh the cache, download the complete repo on an internet-enabled node with: "
        "uv run hf download facebook/sam3 --repo-type model. "
        "You can also set SAM3_MODEL_PATH to the exact local snapshot directory."
    ) from exc

# 2. Open Video
cap = cv2.VideoCapture(str(VIDEO_INPUT_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {VIDEO_INPUT_PATH}")

ret, first_frame = cap.read()
if not ret:
    cap.release()
    raise RuntimeError(
        "Failed to decode first frame from input video. The cluster OpenCV/FFmpeg build likely cannot decode AV1. "
        "Pre-convert input to H.264 (for example with src/rob534/utils.py)."
    )

height, width = first_frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(VIDEO_OUTPUT_PATH), fourcc, fps, (width, height))
if not out.isOpened():
    cap.release()
    raise RuntimeError(f"Failed to open output writer: {VIDEO_OUTPUT_PATH}")

print(f"Processing video with prompt: '{TEXT_PROMPT}'...")

frame_idx = 0
masked_frame_count = 0
previous_mask: np.ndarray | None = None

try:
    frame = first_frame
    while cap.isOpened():
        if frame is None:
            ret, frame = cap.read()
            if not ret:
                break

        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # 3. Inference: Detect and Segment based on text
        inputs = processor(images=pil_img, text=TEXT_PROMPT, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # 4. Post-processing
        # Returns masks for each detected instance of the prompt target
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=SCORE_THRESHOLD,
            mask_threshold=MASK_THRESHOLD,
            target_sizes=[(height, width)]
        )[0]

        # 5. Create Overlay
        # We'll create a single combined mask for all detected instances
        overlay = frame.copy()
        masks = results.get("masks")
        has_masks = masks is not None and len(masks) > 0

        if has_masks:
            candidate_masks = masks.detach().cpu().numpy().astype(bool)
            scores = results.get("scores")
            if scores is None or len(scores) != len(candidate_masks):
                score_values = np.ones(len(candidate_masks), dtype=np.float32)
            else:
                score_values = scores.detach().cpu().numpy()

            if TRACK_SINGLE_INSTANCE:
                if previous_mask is None:
                    selected_idx = int(np.argmax(score_values))
                else:
                    iou_values = np.array([mask_iou(previous_mask, m) for m in candidate_masks], dtype=np.float32)
                    rank_values = score_values + (TRACK_IOU_WEIGHT * iou_values)
                    selected_idx = int(np.argmax(rank_values))
                combined_mask = candidate_masks[selected_idx]
                previous_mask = combined_mask.copy()
            else:
                combined_mask = np.any(candidate_masks, axis=0)
                previous_mask = None

            # Apply a semi-transparent blue tint to the mask area
            mask_color = np.array([255, 100, 0], dtype=np.uint8)  # BGR tint
            overlay[combined_mask] = cv2.addWeighted(
                overlay[combined_mask], 0.5,
                np.full_like(overlay[combined_mask], mask_color), 0.5, 0
            )
            masked_frame_count += 1
        else:
            previous_mask = None

        if frame_idx % 30 == 0:
            if PRINT_DEBUG:
                debug_results = processor.post_process_object_detection(
                    outputs,
                    threshold=0.0,
                    target_sizes=[(height, width)],
                )[0]
                debug_scores = debug_results.get("scores")
                max_score = float(debug_scores.max().item()) if debug_scores is not None and len(debug_scores) > 0 else 0.0
                kept = 0 if masks is None else len(masks)
                total = 0 if debug_scores is None else len(debug_scores)
                print(
                    f"frame={frame_idx} masks={kept} candidates={total} "
                    f"max_score={max_score:.4f} tracked_single={TRACK_SINGLE_INSTANCE}"
                )
            else:
                instance_count = 0 if masks is None else len(masks)
                print(f"frame={frame_idx} masks={instance_count}")

        out.write(overlay)
        frame_idx += 1
        frame = None

except KeyboardInterrupt:
    print("Interrupted by user; finalizing partial video output...")
finally:
    cap.release()
    out.release()
    print(
        f"Finished. wrote_frames={frame_idx}, masked_frames={masked_frame_count}, "
        f"output={VIDEO_OUTPUT_PATH}"
    )