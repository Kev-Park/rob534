"""Run analyze_rollout on every clip in episodes_split/ and save all results to one JSON file.

Usage:
    export GEMINI_API_KEY=...
    python analyze_all_episodes.py
    python analyze_all_episodes.py --dir episodes_split --out results.json
"""
import argparse
import json
from pathlib import Path

from analyze_rollout import analyze_video


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="episodes_split")
    p.add_argument("--out", default="episode_outcomes.json")
    p.add_argument("--model", default="gemini-2.5-pro")
    p.add_argument("--fps", type=float, default=1.0)
    args = p.parse_args()

    videos = sorted(Path(args.dir).glob("*.mp4"))
    print(f"Found {len(videos)} clips in {args.dir}/")

    results = {}
    for v in videos:
        print(f"\n--- {v.name} ---")
        try:
            results[v.name] = analyze_video(v, model=args.model, fps=args.fps)
        except Exception as e:
            print(f"  error: {e}")
            results[v.name] = {"error": str(e)}
        Path(args.out).write_text(json.dumps(results, indent=2))

    print(f"\nDone. Wrote {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
