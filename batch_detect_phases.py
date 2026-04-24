"""
Detect pickup and drop frames for every episode using the gripper position signal.

Pickup = first frame where gripper closes (crosses GRIP_THRESHOLD going up).
Drop   = first frame where gripper opens  (crosses GRIP_THRESHOLD going down)
         after the pickup.

Output: outputs/episode_phases.parquet  (columns: episode_index, pickup_frame,
        drop_frame, pickup_time_s, drop_time_s, n_frames, hold_frames)

Usage:
    python batch_detect_phases.py
    python batch_detect_phases.py --threshold 20 --out outputs/my_phases.parquet
    python batch_detect_phases.py --plot          # show per-episode gripper plots
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── config ────────────────────────────────────────────────────────────────────
ORIG_ROOT = Path(
    r"C:\Users\calle\.cache\huggingface\hub"
    r"\datasets--nc8304--so101_combined_cubeONLY"
    r"\snapshots\acc242c231f60171a5b2833442d176cd793ea8c9"
)
FPS = 30.0
GRIPPER_COL_IDX = 5        # index inside observation.state array
DEFAULT_THRESHOLD = 20.0   # degrees; below = open, above = closed/gripping
DEFAULT_OUT = Path(__file__).parent.parent / "outputs" / "episode_phases.parquet"


# ── detection ─────────────────────────────────────────────────────────────────

def detect_phases(gripper: np.ndarray, threshold: float) -> tuple[int | None, int | None]:
    """
    Returns (pickup_frame, drop_frame) as indices into the episode's frame array.

    The gripper signal has open→close→open cycles.  The robot may make
    multiple gripping attempts before a successful pick-and-place.
    We enumerate all cycles and return the one with the LONGEST hold duration
    (close→open interval), which corresponds to the successful grasp.

    pickup_frame: DOWN crossing of the longest cycle.
    drop_frame  : next UP crossing after that DOWN.
    """
    n = len(gripper)

    ups   = [i for i in range(1, n) if gripper[i - 1] < threshold <= gripper[i]]
    downs = [i for i in range(1, n) if gripper[i - 1] >= threshold > gripper[i]]

    if not ups or not downs:
        return None, None

    best_pickup = best_drop = None
    best_hold   = -1

    for down in downs:
        # must have at least one UP before this close (approach)
        if not any(u < down for u in ups):
            continue
        # drop = next UP after this close
        next_ups = [u for u in ups if u > down]
        if not next_ups:
            continue
        drop  = next_ups[0]
        hold  = drop - down
        if hold > best_hold:
            best_hold   = hold
            best_pickup = down
            best_drop   = drop

    return best_pickup, best_drop


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch detect pickup/drop frames from gripper signal")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Gripper position threshold in degrees (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                        help="Output parquet path")
    parser.add_argument("--plot", action="store_true",
                        help="Show gripper plot for each episode")
    args = parser.parse_args()

    data_path = ORIG_ROOT / "data" / "chunk-000" / "file-000.parquet"
    print(f"Loading: {data_path}")
    df = pd.read_parquet(data_path)

    episodes = sorted(df["episode_index"].unique())
    print(f"Episodes: {len(episodes)}  |  threshold: {args.threshold} deg\n")

    rows = []
    failed = []

    for ep_idx in episodes:
        ep_df = df[df["episode_index"] == ep_idx].sort_values("frame_index")
        states = np.stack(ep_df["observation.state"].values)
        gripper = states[:, GRIPPER_COL_IDX]
        n_frames = len(gripper)

        pf, df_ = detect_phases(gripper, args.threshold)

        if pf is None:
            print(f"  ep {ep_idx:3d}: [FAIL] no pickup detected  "
                  f"(gripper range: {gripper.min():.1f}–{gripper.max():.1f})")
            failed.append(ep_idx)
            rows.append(dict(episode_index=ep_idx,
                             pickup_frame=None, drop_frame=None,
                             pickup_time_s=None, drop_time_s=None,
                             n_frames=n_frames, hold_frames=None))
            continue

        hold_frames = (df_ - pf) if df_ is not None else None
        status = "OK" if df_ is not None else "no-drop"
        print(f"  ep {ep_idx:3d}: pickup={pf:4d} ({pf/FPS:.2f}s)  "
              f"drop={str(df_):>5} ({(df_/FPS if df_ else 0):.2f}s)  "
              f"hold={hold_frames}f  [{status}]")

        rows.append(dict(
            episode_index=ep_idx,
            pickup_frame=pf,
            drop_frame=df_,
            pickup_time_s=round(pf / FPS, 3),
            drop_time_s=round(df_ / FPS, 3) if df_ is not None else None,
            n_frames=n_frames,
            hold_frames=hold_frames,
        ))

        if args.plot:
            _plot_episode(gripper, pf, df_, ep_idx, args.threshold)

    result = pd.DataFrame(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    ok = result["pickup_frame"].notna().sum()
    has_drop = result["drop_frame"].notna().sum()
    print(f"Summary: {ok}/{len(episodes)} episodes have pickup  |  "
          f"{has_drop}/{len(episodes)} have both pickup+drop")
    if failed:
        print(f"Failed episodes: {failed}")


def _plot_episode(gripper: np.ndarray, pickup: int | None,
                  drop: int | None, ep_idx: int, threshold: float):
    import matplotlib.pyplot as plt
    t = np.arange(len(gripper)) / FPS
    plt.figure(figsize=(12, 3))
    plt.plot(t, gripper, color="steelblue", linewidth=0.8, label="gripper.pos")
    plt.axhline(threshold, color="orange", linestyle="--", linewidth=1, label=f"threshold={threshold}")
    if pickup is not None:
        plt.axvline(pickup / FPS, color="lime", linewidth=1.5, label=f"pickup f{pickup}")
    if drop is not None:
        plt.axvline(drop / FPS, color="tomato", linewidth=1.5, label=f"drop f{drop}")
    plt.title(f"Episode {ep_idx} — gripper position")
    plt.xlabel("time (s)")
    plt.ylabel("gripper.pos (deg)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
