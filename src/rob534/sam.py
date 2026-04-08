import cv2
import torch
import numpy as np
import os
from pathlib import Path
from PIL import Image
from transformers import Sam3Processor, Sam3Model

# --- Configuration ---
TEXT_PROMPT = "the block gripped by the gripper"
VIDEO_INPUT = "file-000-h264.mp4"
VIDEO_OUTPUT = "file-000-segmented.mp4"
MASK_THRESHOLD = 0.3
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
            threshold=MASK_THRESHOLD,
            target_sizes=[(height, width)]
        )[0]

        # 5. Create Overlay
        # We'll create a single combined mask for all detected instances
        overlay = frame.copy()
        masks = results.get("masks")
        has_masks = masks is not None and len(masks) > 0

        if has_masks:
            # Combine all instance masks into one boolean mask
            combined_mask = torch.any(masks, dim=0).cpu().numpy().astype(bool)

            # Apply a semi-transparent blue tint to the mask area
            mask_color = np.array([255, 100, 0], dtype=np.uint8)  # BGR tint
            overlay[combined_mask] = cv2.addWeighted(
                overlay[combined_mask], 0.5,
                np.full_like(overlay[combined_mask], mask_color), 0.5, 0
            )
            masked_frame_count += 1

        if frame_idx % 30 == 0:
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