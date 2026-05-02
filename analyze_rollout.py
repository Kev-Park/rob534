"""Read a video from ./rollouts/ and classify the robot arm failure mode with Gemini Flash.

Usage:
    python analyze_rollout.py                               # analyze all videos in rollouts/
    python analyze_rollout.py rollouts/rollout_XYZ.mp4      # analyze a specific file
    python analyze_rollout.py --model gemini-2.5-flash
    python analyze_rollout.py --no-open                     # skip opening video in player

Requirements:
    pip install google-genai pandas
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import cv2
import pandas as pd
from huggingface_hub import snapshot_download

from google import genai
from google.genai import types

PICK_QUALITIES = ["clean", "messy", "failed"]
DROP_QUALITIES = ["clean", "clean_with_correction", "messy_with_correction", "failed", "not_attempted"]

DESCRIBE_PROMPT = """You are watching a video of a robotic arm doing a pick-and-place task.
Describe exactly what you see in plain, factual language. Cover these three phases in order:

1. PICK: How does the arm approach the block? Does it grasp it on the first attempt?
   Does the block move or rotate in the gripper? Does the grip look stable?

2. TRANSPORT: Once grasped (if it is), how does the arm carry the block?
   Does it move smoothly or erratically? Does the block stay secure?

3. DROP: Does the arm arrive over the target region? Does it release the block?
   Does the block land on target? Any corrections or retries before release?

End with one sentence on the final state: where is the block at the end of the video?

Be specific and concrete — describe what you actually see, not what you expect.
Do NOT classify yet. Just describe.
"""

CLASSIFY_PROMPT = """Based on the following description of a robotic arm pick-and-place attempt,
classify the episode using these exact categories:

PICK QUALITY:
  clean                  — smooth approach, grasped in one attempt, stable grip
  messy                  — grasped but multiple attempts, knocked block first, or unstable grip
  failed                 — never achieved a lasting grasp

DROP QUALITY:
  clean                  — arrived over target and released smoothly in one motion
  clean_with_correction  — made a positional correction before a clean release onto target
  messy_with_correction  — corrected but release was imperfect (bounced, partial, forceful)
  failed                 — reached target area but never released, or released off-target
  not_attempted          — pick failed so no meaningful drop phase occurred

overall_success: true if the block ends up on the correct target region, false otherwise.
confidence: your confidence in the classification (0–1).
reasoning: one sentence explaining the key evidence from the description.

Description:
{description}

Return ONLY a JSON object with fields: pick_quality, drop_quality, overall_success, confidence, reasoning.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "pick_quality":    {"type": "string", "enum": PICK_QUALITIES},
        "drop_quality":    {"type": "string", "enum": DROP_QUALITIES},
        "overall_success": {"type": "boolean"},
        "confidence":      {"type": "number"},
        "reasoning":       {"type": "string"},
    },
    "required": ["pick_quality", "drop_quality", "overall_success", "confidence", "reasoning"],
}


def analyze_rollout(dataset_id: str) -> pd.DataFrame:
    """Load a LeRobot-format HuggingFace dataset and return the data as a DataFrame.

    Args:
        dataset_id: HuggingFace dataset ID, e.g. "nc8304/eval_smolvla-phase-split_combined"

    Returns:
        pd.DataFrame with one row per timestep from the data/ parquets.
    """
    repo_dir = snapshot_download(repo_id=dataset_id, repo_type="dataset")
    parquet_files = sorted(glob.glob(f"{repo_dir}/data/**/*.parquet", recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {repo_dir}/data/")
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    return df


def all_rollouts(folder: str = "rollouts") -> list[Path]:
    """Return all .mp4 files in `folder`, sorted by name."""
    p = Path(folder)
    if not p.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}/")
    videos = sorted(p.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No .mp4 files in {folder}/")
    return videos


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


def load_api_key(key_file: str | None) -> str:
    if key_file:
        key = Path(key_file).read_text().strip()
        if not key:
            raise RuntimeError(f"API key file is empty: {key_file}")
        return key
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment or pass --api-key-file.")
    return key


def _stream_text(client: genai.Client, model: str, contents, config) -> str:
    full_text = ""
    for chunk in client.models.generate_content_stream(
        model=model, contents=contents, config=config
    ):
        if chunk.text:
            print(chunk.text, end="", flush=True)
            full_text += chunk.text
    print()
    return full_text


def analyze_video(video_path: Path, model: str = "gemini-2.5-flash", open_player: bool = True, key_file: str | None = None) -> dict:
    api_key = load_api_key(key_file)
    client = genai.Client(api_key=api_key)

    print(f"Uploading {video_path} to Gemini Files API...")
    uploaded = client.files.upload(file=str(video_path))
    uploaded = wait_for_file_active(client, uploaded)

    if open_player:
        try:
            subprocess.Popen(["cmd", "/c", "start", "", str(video_path.resolve())])
        except Exception as e:
            print(f"(note: could not open video player: {e})")

    try:
        # Pass 1 — describe what happens in the video
        print(f"\n[pass 1] Describing video with {model}...\n")
        description = _stream_text(
            client, model,
            contents=[uploaded, DESCRIBE_PROMPT],
            config=types.GenerateContentConfig(temperature=0.4),
        )

        # Pass 2 — classify from the description (text only, no video)
        print(f"\n[pass 2] Classifying description...\n")
        classify_prompt = CLASSIFY_PROMPT.format(description=description)
        result_text = _stream_text(
            client, model,
            contents=[classify_prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                temperature=0.2,
            ),
        )
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception as e:
            print(f"(note: could not delete uploaded file {uploaded.name}: {e})")

    try:
        result = json.loads(result_text)
        result["description"] = description  # carry description forward for overlay/CSV
        return result
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse model response: {result_text!r}") from e


def extract_episode_clip(video_path: Path, from_ts: float, to_ts: float, out_path: Path) -> Path:
    """Extract a time segment from a video using ffmpeg (re-encode for frame accuracy)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = to_ts - from_ts
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(from_ts),   # fast input seek to nearest keyframe
        "-i", str(video_path),
        "-t", str(duration),   # exact duration after seek
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an",                 # no audio
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr}")
    return out_path


def analyze_and_overlay_dataset(
    dataset_id: str,
    key_file: str | None = None,
    model: str = "gemini-2.5-flash",
    output_dir: str = "labeled_rollouts",
    max_episodes: int | None = None,
) -> pd.DataFrame:
    """Download a LeRobot dataset, run Gemini on each episode, and overlay results.

    For each episode:
      1. Extracts the clip from the shared video using ffmpeg.
      2. Sends it to Gemini for failure-mode classification.
      3. Burns the result into the top-left corner of the clip.

    Returns a DataFrame with one row per episode.
    """
    repo_dir = Path(snapshot_download(repo_id=dataset_id, repo_type="dataset"))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_files = sorted((repo_dir / "meta" / "episodes").glob("**/*.parquet"))
    if not meta_files:
        raise FileNotFoundError(f"No meta/episodes parquets found in {repo_dir}")
    episodes_df = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)

    cam_key = "observation.images.camera1"
    video_dir = repo_dir / "videos" / cam_key

    rows = []
    total = len(episodes_df)
    if max_episodes is not None:
        episodes_df = episodes_df.head(max_episodes)
    for _, ep in episodes_df.iterrows():
        ep_idx = int(ep["episode_index"])
        chunk = int(ep[f"videos/{cam_key}/chunk_index"])
        file_i = int(ep[f"videos/{cam_key}/file_index"])
        from_ts = float(ep[f"videos/{cam_key}/from_timestamp"])
        to_ts = float(ep[f"videos/{cam_key}/to_timestamp"])

        video_path = video_dir / f"chunk-{chunk:03d}" / f"file-{file_i:03d}.mp4"
        if not video_path.exists():
            print(f"[{ep_idx+1}/{total}] Video not found: {video_path}", file=sys.stderr)
            rows.append({"episode_index": ep_idx, "failure_mode": "ERROR", "confidence": None, "reasoning": "video not found"})
            continue

        clip_path = out_dir / f"episode_{ep_idx:04d}_clip.mp4"
        print(f"\n[{ep_idx+1}/{total}] Extracting episode {ep_idx} ({from_ts:.2f}s – {to_ts:.2f}s)")
        extract_episode_clip(video_path, from_ts, to_ts, clip_path)

        print(f"[{ep_idx+1}/{total}] Analyzing with Gemini...")
        try:
            result = analyze_video(clip_path, model=model, open_player=False, key_file=key_file)
            overlay_result(clip_path, result)
            rows.append({"episode_index": ep_idx, **result})
        except Exception as e:
            print(f"ERROR on episode {ep_idx}: {e}", file=sys.stderr)
            rows.append({"episode_index": ep_idx, "pick_quality": "ERROR", "drop_quality": "ERROR", "overall_success": False, "confidence": None, "reasoning": str(e)})
        finally:
            clip_path.unlink(missing_ok=True)  # remove unlabeled clip

    df = pd.DataFrame(rows, columns=["episode_index", "pick_quality", "drop_quality", "overall_success", "confidence", "reasoning", "description"])
    return df


def overlay_result(video_path: Path, result: dict) -> Path:
    """Burn the Gemini result into the top-left corner of every frame.

    Returns the path to the new video (same name with '_labeled' suffix).
    """
    out_path = video_path.with_stem(video_path.stem + "_labeled")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_h = 22
    pad = 8

    # Build overlay lines
    success = "OK" if result.get("overall_success") else "FAIL"
    header = (
        f"pick={result.get('pick_quality','?')}  "
        f"drop={result.get('drop_quality','?')}  "
        f"{success}  (conf: {result.get('confidence', 0):.2f})"
    )
    wrapped = textwrap.wrap(result.get("reasoning", ""), width=55)
    lines = [header] + wrapped

    # Background box height
    box_h = pad * 2 + line_h * len(lines)
    # Max text width (approximate)
    box_w = max(cv2.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines) + pad * 2

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, line in enumerate(lines):
            y = pad + (i + 1) * line_h
            # Thin white shadow for legibility
            cv2.putText(frame, line, (pad + 1, y + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(frame, line, (pad, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Labeled video saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video", nargs="?", default=None,
        help="Path to a specific video. If omitted, analyzes all .mp4s in --rollouts-dir."
    )
    parser.add_argument("--rollouts-dir", default="rollouts")
    parser.add_argument("--dataset", default="nc8304/eval_smolvla-phase-split_combined",
                        help="HuggingFace dataset ID. Overrides --rollouts-dir and positional video argument.")
    parser.add_argument("--output-dir", default="labeled_rollouts",
                        help="Directory for labeled clips when using --dataset.")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--max-episodes", type=int, default=10, help="Stop after this many episodes (default: 10, 0 = all).")
    parser.add_argument("--no-open", action="store_true", help="Don't open videos in a player.")
    parser.add_argument("--api-key-file", default=r"C:\Users\calle\Desktop\gem.txt", help="Path to a text file containing the Gemini API key.")
    parser.add_argument("--output", default="results.csv", help="CSV file to save results (default: results.csv).")
    args = parser.parse_args()

    if args.dataset:
        df = analyze_and_overlay_dataset(
            args.dataset,
            key_file=args.api_key_file,
            model=args.model,
            output_dir=args.output_dir,
            max_episodes=args.max_episodes if args.max_episodes > 0 else None,
        )
        print("\n=== Results ===")
        print(df.to_string(index=False))
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
        return

    videos = [Path(args.video)] if args.video else all_rollouts(args.rollouts_dir)

    rows = []
    for i, video_path in enumerate(videos):
        if not video_path.exists():
            print(f"Video not found: {video_path}", file=sys.stderr)
            continue
        print(f"\n[{i+1}/{len(videos)}] Analyzing: {video_path.name}")
        try:
            result = analyze_video(video_path, model=args.model, open_player=not args.no_open, key_file=args.api_key_file)
            overlay_result(video_path, result)
            rows.append({"video": video_path.name, **result})
        except Exception as e:
            print(f"ERROR on {video_path.name}: {e}", file=sys.stderr)
            rows.append({"video": video_path.name, "pick_quality": "ERROR", "drop_quality": "ERROR", "overall_success": False, "confidence": None, "reasoning": str(e)})

    df = pd.DataFrame(rows, columns=["video", "pick_quality", "drop_quality", "overall_success", "confidence", "reasoning", "description"])
    print("\n=== Results ===")
    print(df.to_string(index=False))
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
