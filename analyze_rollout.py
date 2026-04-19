"""Classify robot-arm failure modes in rollout videos with Gemini Flash.

Usage:
    export GEMINI_API_KEY=...

    # single video (defaults to newest mp4 in rollouts/)
    python analyze_rollout.py
    python analyze_rollout.py rollouts/rollout_XYZ.mp4

    # batch: every mp4 in rollouts/, write per-video + summary to ./analysis/
    python analyze_rollout.py --all
    python analyze_rollout.py --all --limit 20 --workers 4
    python analyze_rollout.py --all --output-dir analysis --pattern "rollout_2026*.mp4"

Requirements:
    pip install google-genai
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

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


@dataclass
class Result:
    video: str
    failure_mode: str | None
    confidence: float | None
    reasoning: str | None
    error: str | None = None


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.")
    return genai.Client(api_key=api_key)


def latest_rollout(folder: str = "rollouts") -> Path:
    p = Path(folder)
    if not p.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}/")
    videos = sorted(p.glob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not videos:
        raise FileNotFoundError(f"No .mp4 files in {folder}/")
    return videos[0]


def wait_for_file_active(client: genai.Client, file_obj, timeout: float = 180.0):
    deadline = time.time() + timeout
    while file_obj.state.name == "PROCESSING":
        if time.time() > deadline:
            raise TimeoutError(f"File {file_obj.name} still PROCESSING after {timeout}s")
        time.sleep(2)
        file_obj = client.files.get(name=file_obj.name)
    if file_obj.state.name != "ACTIVE":
        raise RuntimeError(f"File upload ended in state {file_obj.state.name}")
    return file_obj


def analyze_video(video_path: Path, client: genai.Client, model: str) -> dict:
    uploaded = client.files.upload(file=str(video_path))
    uploaded = wait_for_file_active(client, uploaded)
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
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass
    return json.loads(response.text)


def analyze_one(video_path: Path, client: genai.Client, model: str) -> Result:
    try:
        data = analyze_video(video_path, client, model)
        return Result(
            video=str(video_path),
            failure_mode=data.get("failure_mode"),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning"),
        )
    except Exception as e:
        return Result(video=str(video_path), failure_mode=None,
                      confidence=None, reasoning=None, error=str(e))


def summarize(results: list[Result]) -> dict:
    ok = [r for r in results if r.error is None and r.failure_mode]
    failed = [r for r in results if r.error is not None]
    counts = Counter(r.failure_mode for r in ok)

    per_mode = {}
    for mode in FAILURE_MODES:
        rs = [r for r in ok if r.failure_mode == mode]
        per_mode[mode] = {
            "count": len(rs),
            "fraction": (len(rs) / len(ok)) if ok else 0.0,
            "mean_confidence": mean(r.confidence for r in rs) if rs else None,
        }

    return {
        "total_videos": len(results),
        "classified": len(ok),
        "errored": len(failed),
        "mean_confidence_overall": mean(r.confidence for r in ok) if ok else None,
        "counts": dict(counts),
        "per_failure_mode": per_mode,
        "errors": [{"video": r.video, "error": r.error} for r in failed],
    }


def print_summary(summary: dict) -> None:
    print("\n=== Batch summary ===")
    print(f"Videos analyzed: {summary['classified']}/{summary['total_videos']}"
          f" (errored: {summary['errored']})")
    mc = summary["mean_confidence_overall"]
    print(f"Mean confidence: {mc:.2f}" if mc is not None else "Mean confidence: n/a")
    print("\nFailure mode distribution:")
    for mode, stats in summary["per_failure_mode"].items():
        conf = f"{stats['mean_confidence']:.2f}" if stats["mean_confidence"] is not None else "n/a"
        print(f"  {mode:45s}  {stats['count']:3d}  ({stats['fraction']*100:5.1f}%)  conf={conf}")
    if summary["errors"]:
        print("\nErrors:")
        for e in summary["errors"]:
            print(f"  {e['video']}: {e['error']}")


def write_outputs(results: list[Result], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    results_path = out_dir / f"results_{stamp}.json"
    with results_path.open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    csv_path = out_dir / f"results_{stamp}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "failure_mode", "confidence", "reasoning", "error"])
        for r in results:
            w.writerow([r.video, r.failure_mode or "", r.confidence if r.confidence is not None else "",
                        r.reasoning or "", r.error or ""])

    summary_path = out_dir / f"summary_{stamp}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote: {results_path}\n       {csv_path}\n       {summary_path}")


def collect_videos(folder: str, pattern: str, limit: int | None) -> list[Path]:
    p = Path(folder)
    if not p.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}/")
    videos = sorted(p.glob(pattern), key=lambda f: f.stat().st_mtime)
    if not videos:
        raise FileNotFoundError(f"No videos matching {pattern} in {folder}/")
    return videos[:limit] if limit else videos


def run_batch(videos: list[Path], model: str, workers: int) -> list[Result]:
    client = _client()
    results: list[Result] = []
    total = len(videos)

    if workers <= 1:
        for i, v in enumerate(videos, 1):
            print(f"[{i}/{total}] {v.name}")
            results.append(analyze_one(v, client, model))
        return results

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(analyze_one, v, client, model): v for v in videos}
        done = 0
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            tag = r.failure_mode or f"ERROR: {r.error}"
            print(f"[{done}/{total}] {Path(r.video).name} -> {tag}")
            results.append(r)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default=None,
                        help="Single video path. Ignored if --all is set.")
    parser.add_argument("--all", action="store_true",
                        help="Process every matching video in --rollouts-dir.")
    parser.add_argument("--rollouts-dir", default="rollouts")
    parser.add_argument("--pattern", default="*.mp4",
                        help="Glob within rollouts-dir when --all is used (default: *.mp4).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max videos to process in batch mode.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel Gemini calls in batch mode (start with 2-4).")
    parser.add_argument("--output-dir", default="analysis",
                        help="Where to write batch results + summary JSON/CSV.")
    parser.add_argument("--model", default="gemini-2.5-flash")
    args = parser.parse_args()

    if args.all:
        videos = collect_videos(args.rollouts_dir, args.pattern, args.limit)
        print(f"Analyzing {len(videos)} videos with {args.model} (workers={args.workers})")
        results = run_batch(videos, args.model, args.workers)
        summary = summarize(results)
        print_summary(summary)
        write_outputs(results, summary, Path(args.output_dir))
        return

    video_path = Path(args.video) if args.video else latest_rollout(args.rollouts_dir)
    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing: {video_path}")
    result = analyze_video(video_path, _client(), args.model)
    print("\n=== Result ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
