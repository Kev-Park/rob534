"""Read a video from ./rollouts/ and classify the rollout outcome with Gemini Pro.

Usage:
    export GEMINI_API_KEY=...
    python analyze_rollout.py                               # uses most recent video in rollouts/
    python analyze_rollout.py rollouts/rollout_XYZ.mp4      # analyze a specific file
    python analyze_rollout.py --model gemini-2.5-pro

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


OUTCOMES = [
    "success",
    "failure_to_pick_up_block",
    "failure_to_move_block_over_correct_region",
    "failure_to_drop_block",
]

PROMPT = """You are reviewing a video of a robotic arm attempting a pick-and-place task.
The task is to pick up a cube and drop it into a cube-shaped hole in the target region.
The rollout is only a success if the cube actually FALLS THROUGH the hole — landing on
top of the hole, next to it, or bouncing off the edge does NOT count as success.

Classify the rollout into exactly ONE of the following outcomes:

1. success
   The arm grasped the cube, carried it over the hole, released it, and the cube
   visibly fell THROUGH the cube-shaped hole (disappearing into / below the target).
   If the cube lands on top of, beside, or bounces off the hole, this is NOT success.

2. failure_to_pick_up_block
   The arm never got a solid, stable grasp of the cube. This includes:
     - Missing the cube entirely or closing the gripper on empty air
     - Just nudging, knocking, or brushing the cube without lifting it
     - The cube slipping out of the gripper immediately or within the first ~1s of lifting
     - Only partially gripping it so the cube falls back to the table near the pickup point
   IMPORTANT: if the cube ever falls out of the gripper before the arm has clearly
   transported it across to the target area, classify this as failure_to_pick_up_block,
   NOT failure_to_move_block_over_correct_region. A drop near the pickup zone is a
   pickup failure, not a transport failure.

3. failure_to_move_block_over_correct_region
   The arm got a SOLID grasp of the cube and held it stably while moving, but failed
   to bring it over the cube-shaped hole. Use this only when the gripper clearly
   transports the cube some distance and the failure is about WHERE it goes — wrong
   target, stops short, overshoots, or the cube only escapes the gripper near/over
   the wrong region after sustained transport.

4. failure_to_drop_block
   The arm grasped the cube, carried it cleanly over the hole, but failed to release
   it (gripper stayed closed) OR released it but the cube landed on top of / beside
   the hole instead of falling through.

Decision order:
  - First check pickup: was the grasp solid and sustained? If no → failure_to_pick_up_block.
  - Then check transport: did it reach the hole? If no → failure_to_move_block_over_correct_region.
  - Then check release: did the cube actually fall through the hole? If yes → success,
    otherwise → failure_to_drop_block.

Return ONLY a JSON object with fields: outcome, confidence (0-1), reasoning (1-3 sentences).
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "outcome": {"type": "string", "enum": OUTCOMES},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["outcome", "confidence", "reasoning"],
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


def analyze_video(video_path: Path, model: str = "gemini-2.5-pro", fps: float = 5.0) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.")

    client = genai.Client(api_key=api_key)

    print(f"Uploading {video_path} to Gemini Files API...")
    uploaded = client.files.upload(file=str(video_path))
    uploaded = wait_for_file_active(client, uploaded)

    # Wrap the file as a Part so we can override the default 1 fps sampling rate.
    video_part = types.Part(
        file_data=types.FileData(file_uri=uploaded.uri, mime_type=uploaded.mime_type),
        video_metadata=types.VideoMetadata(fps=fps),
    )

    print(f"Calling {model} (sampling at {fps} fps)...")
    try:
        response = client.models.generate_content(
            model=model,
            contents=[video_part, PROMPT],
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
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Frame sampling rate sent to Gemini (default 5.0; Gemini default is 1.0)")
    args = parser.parse_args()

    video_path = Path(args.video) if args.video else latest_rollout(args.rollouts_dir)
    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing: {video_path}")
    result = analyze_video(video_path, model=args.model, fps=args.fps)

    print("\n=== Result ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
