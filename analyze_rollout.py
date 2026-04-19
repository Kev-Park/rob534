"""Read a video from ./rollouts/ and classify the robot arm failure mode with Gemini Flash.

Usage:
    export GEMINI_API_KEY=...
    python analyze_rollout.py                               # uses most recent video in rollouts/
    python analyze_rollout.py rollouts/rollout_XYZ.mp4      # analyze a specific file
    python analyze_rollout.py --model gemini-2.5-flash

Requirements:
    pip install google-genai
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types


FAILURE_MODES = [
    "failure_to_pick_up_block",
    "failure_to_move_block_over_correct_region",
    "failure_to_drop_block",
]

PROMPT = """You are reviewing a video of a robotic arm attempting a pick-and-place task:
pick up a block, move it over the correct target region, and drop it there.

The rollout failed. Identify which ONE of the following failure modes best matches what you see:

1. failure_to_pick_up_block
   The arm never successfully grasped the block. It may have missed, knocked the block,
   closed the gripper on empty air, or dropped the block immediately at the pickup point.

2. failure_to_move_block_over_correct_region
   The arm did grasp the block, but failed to bring it over the correct target region.
   It moved to the wrong place, stopped short, or dropped the block in transit.

3. failure_to_drop_block
   The arm grasped the block AND moved it over the correct region, but failed to release it.
   The gripper never opened, opened too late, or the block stayed stuck to the gripper.

Classify strictly — pick the earliest failure in the sequence pickup -> transport -> release.
Return ONLY a JSON object with fields: failure_mode, confidence (0-1), reasoning (1-3 sentences).
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "failure_mode": {"type": "string", "enum": FAILURE_MODES},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["failure_mode", "confidence", "reasoning"],
}


def latest_rollout(folder: str = "rollouts") -> Path:
    """Return the most recently modified .mp4 in `folder`."""
    p = Path(folder)
    if not p.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}/")
    videos = sorted(p.glob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not videos:
        raise FileNotFoundError(f"No .mp4 files in {folder}/")
    return videos[0]


def wait_for_file_active(client: genai.Client, file_obj, timeout: float = 120.0):
    """Poll the Files API until the uploaded video is ACTIVE (or fail/timeout)."""
    deadline = time.time() + timeout
    while file_obj.state.name == "PROCESSING":
        if time.time() > deadline:
            raise TimeoutError(f"File {file_obj.name} still PROCESSING after {timeout}s")
        time.sleep(2)
        file_obj = client.files.get(name=file_obj.name)
    if file_obj.state.name != "ACTIVE":
        raise RuntimeError(f"File upload ended in state {file_obj.state.name}: {file_obj}")
    return file_obj


def analyze_video(video_path: Path, model: str = "gemini-2.5-flash") -> dict:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.")

    client = genai.Client(api_key=api_key)

    print(f"Uploading {video_path} to Gemini Files API...")
    uploaded = client.files.upload(file=str(video_path))
    uploaded = wait_for_file_active(client, uploaded)

    print(f"Calling {model}...")
    try:
        response = client.models.generate_content(
            model=model,
            contents=[uploaded, PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                temperature=0.1,
            ),
        )
    finally:
        # Clean up the uploaded file regardless of outcome.
        try:
            client.files.delete(name=uploaded.name)
        except Exception as e:
            print(f"(note: could not delete uploaded file {uploaded.name}: {e})")

    return json.loads(response.text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video", nargs="?", default=None,
        help="Path to video. If omitted, picks the newest .mp4 in rollouts/."
    )
    parser.add_argument("--rollouts-dir", default="rollouts")
    parser.add_argument("--model", default="gemini-2.5-flash")
    args = parser.parse_args()

    video_path = Path(args.video) if args.video else latest_rollout(args.rollouts_dir)
    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing: {video_path}")
    result = analyze_video(video_path, model=args.model)

    print("\n=== Result ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
