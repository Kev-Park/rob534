"""
phase_episode_sampler.py
────────────────────────
Generates training windows from colour-augmented episodes using
the pre-computed phase labels (pickup / hold / drop frames).

For each episode three sub-trajectories are produced, all starting
at frame 0 but ending at a different phase boundary:

    ┌─────────────────────────────────────────────────────────┐
    │ frame 0                                         n_frames │
    │ ├──── approach ────┤                                      │
    │ ├──────────── carry ───────────────┤                      │
    │ ├──────────────────── full ──────────────────────────────┤│
    │                   ↑              ↑                        │
    │              pickup_frame    drop_frame                   │
    └─────────────────────────────────────────────────────────┘

Within each sub-trajectory, a sliding window of `window_size` frames
is stepped forward by `stride = window_size * (1 - overlap)` frames
so every important transition appears in multiple training samples.

Basic Usage (video frame indices only)
───────────────────────────────────────
    from phase_episode_sampler import iter_phase_windows

    for sample in iter_phase_windows("outputs/episode_phases.csv",
                                      window_size=50, overlap=0.25):
        ep     = sample["episode_index"]   # int  – which episode
        frames = sample["frames"]          # list – frame indices in this window
        seg    = sample["segment"]         # str  – 'approach' | 'carry' | 'full'
        # load frames[0]…frames[-1] from your dataset and feed to the model

With Motor Data
───────────────
Pass the data_df DataFrame (from load_meta()) to get the robot joint
positions and actions for every frame in each window:

    from episode_labeler import load_meta
    from phase_episode_sampler import iter_phase_windows

    _, data_df = load_meta()   # loads the full LeRobot parquet

    for sample in iter_phase_windows(window_size=50, overlap=0.25,
                                      data_df=data_df):
        ep     = sample["episode_index"]
        frames = sample["frames"]
        seg    = sample["segment"]
        motor  = sample["motor_data"]   # dict, or None if data_df not supplied

        state  = motor["observation.state"]  # shape (window_size, 6)  – joint °
        action = motor["action"]             # shape (window_size, 6)  – commanded
        # axis 1 layout: [joint0, joint1, joint2, joint3, joint4, gripper]
        #   index 5 = gripper

Choose which motor columns to pull with motor_keys (default: both):
    for sample in iter_phase_windows(data_df=data_df,
                                      motor_keys=("observation.state",)):
        state = sample["motor_data"]["observation.state"]  # shape (window_size, 6)
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── default path to the phase CSV produced by batch_detect_phases.py ──────────
DEFAULT_PHASES_CSV = Path(__file__).parent.parent / "outputs" / "episode_phases.csv"


def iter_phase_windows(
    phases_csv: str | Path = DEFAULT_PHASES_CSV,
    window_size: int = 50,
    overlap: float = 0.25,
    segments: tuple[str, ...] = ("approach", "carry", "full"),
    skip_missing: bool = True,
    data_df: pd.DataFrame | None = None,
    motor_keys: tuple[str, ...] = ("observation.state", "action"),
):
    """
    Iterate over overlapping training windows across all episodes.

    Parameters
    ----------
    phases_csv : path to episode_phases.csv (or .parquet)
        The file produced by batch_detect_phases.py.  Contains one row
        per episode with pickup_frame, hold_frame, drop_frame, n_frames.

    window_size : int
        Number of consecutive frames in each training window.
        Tune this to your model's context length / chunk size.

    overlap : float  [0, 1)
        Fraction of window_size that consecutive windows share.
        e.g. 0.25 → stride = 75% of window_size, windows overlap by 25%.
        Higher overlap → more training samples, slower iteration.

    segments : tuple of str
        Which sub-trajectories to generate windows for.
        Any subset of ("approach", "carry", "full").
          "approach" – frame 0 → pickup_frame  (robot reaches for block)
          "carry"    – frame 0 → drop_frame    (reach + grasp + transport)
          "full"     – frame 0 → n_frames      (complete episode + release)

    skip_missing : bool
        If True (default), silently skip episodes where pickup or drop
        was not detected (episodes 4 and 75 in this dataset).
        If False, those episodes raise a ValueError.

    data_df : pd.DataFrame or None
        The data parquet loaded from load_meta() (second return value).
        When provided, each yielded sample includes a "motor_data" dict
        with the robot joint readings for every frame in the window.
        When None (default), "motor_data" is None in every sample.

    motor_keys : tuple of str
        Which columns to extract from data_df.
        Default: ("observation.state", "action")
          "observation.state" – measured joint positions, shape (window_size, 6)
          "action"            – commanded joint positions, shape (window_size, 6)
        Joint axis layout: [joint0, joint1, joint2, joint3, joint4, gripper]
          index 5 = gripper

    Yields
    ------
    dict with keys:
        "episode_index"  int              – episode number (0-based)
        "segment"        str              – 'approach' | 'carry' | 'full'
        "frames"         list[int]        – frame indices [start … start+window_size-1]
        "window_start"   int              – first frame index (inclusive)
        "window_end"     int              – last frame index (exclusive)
        "pickup_frame"   int | None       – pickup frame for this episode
        "drop_frame"     int | None       – drop frame for this episode
        "n_frames"       int              – total episode length
        "motor_data"     dict | None      – motor arrays keyed by column name,
                                           or None if data_df was not supplied.
                                           Each array has shape (window_size, n_joints).
    """

    # ── load phase labels ──────────────────────────────────────────────────────
    path = Path(phases_csv)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # how far to step between consecutive windows
    stride = max(1, int(window_size * (1 - overlap)))

    # ── pre-group motor data by episode for fast per-window lookup ─────────────
    # Grouping once here avoids re-scanning the full DataFrame for every window.
    if data_df is not None:
        ep_data: dict[int, pd.DataFrame] = {
            ep: grp.sort_values("frame_index").reset_index(drop=True)
            for ep, grp in data_df.groupby("episode_index")
        }
    else:
        ep_data = {}

    # ── iterate episodes ───────────────────────────────────────────────────────
    for _, row in df.iterrows():
        ep       = int(row["episode_index"])
        n_frames = int(row["n_frames"])

        # check for missing phase detection
        pickup_missing = pd.isna(row["pickup_frame"])
        drop_missing   = pd.isna(row["drop_frame"])

        if (pickup_missing or drop_missing) and not skip_missing:
            raise ValueError(
                f"Episode {ep} has no detected phases. "
                "Re-run batch_detect_phases.py or set skip_missing=True."
            )

        if pickup_missing or drop_missing:
            # can still yield 'full' segment even without phase labels
            pickup_frame = None
            drop_frame   = None
        else:
            pickup_frame = int(row["pickup_frame"])
            drop_frame   = int(row["drop_frame"])

        # ── define the end frame for each sub-trajectory ──────────────────────
        #
        #   approach : 0 → pickup_frame   (how the robot reaches the block)
        #   carry    : 0 → drop_frame     (approach + grasp + transport)
        #   full     : 0 → n_frames       (everything, including release)
        #
        segment_ends = {
            "approach": pickup_frame,   # None if undetected
            "carry":    drop_frame,     # None if undetected
            "full":     n_frames,
        }

        # ── pre-fetch motor rows for this episode (if requested) ──────────────
        ep_rows = ep_data.get(ep)   # sorted DataFrame or None

        # ── slide windows through each requested sub-trajectory ───────────────
        for seg_name in segments:
            seg_end = segment_ends[seg_name]

            # skip this segment if the boundary wasn't detected
            if seg_end is None:
                continue

            # the sub-trajectory must be at least one full window long
            if seg_end < window_size:
                continue

            # slide the window from frame 0 to seg_end
            start = 0
            while start + window_size <= seg_end:

                # ── extract motor data for this window ────────────────────────
                if ep_rows is not None:
                    fi   = ep_rows["frame_index"].values   # sorted int array
                    mask = (fi >= start) & (fi < start + window_size)
                    window_rows = ep_rows[mask]
                    motor_data: dict | None = {}
                    for key in motor_keys:
                        if key not in window_rows.columns:
                            continue
                        vals = window_rows[key].values
                        if len(vals) == 0:
                            motor_data[key] = np.empty((0,))
                        elif isinstance(vals[0], np.ndarray):
                            motor_data[key] = np.stack(vals)   # (window_size, n_joints)
                        else:
                            motor_data[key] = vals.astype(float)
                else:
                    motor_data = None

                yield {
                    "episode_index": ep,
                    "segment":       seg_name,
                    "frames":        list(range(start, start + window_size)),
                    "window_start":  start,
                    "window_end":    start + window_size,
                    "pickup_frame":  pickup_frame,
                    "drop_frame":    drop_frame,
                    "n_frames":      n_frames,
                    "motor_data":    motor_data,
                }
                start += stride


# ── convenience: collect all windows into a DataFrame ─────────────────────────

def phase_windows_dataframe(
    phases_csv: str | Path = DEFAULT_PHASES_CSV,
    window_size: int = 50,
    overlap: float = 0.25,
    segments: tuple[str, ...] = ("approach", "carry", "full"),
) -> pd.DataFrame:
    """
    Same as iter_phase_windows but returns a DataFrame instead of a generator.
    Each row is one training window (without the 'frames' or 'motor_data' columns).

    Useful for inspecting the full sample plan before training.
    For motor data, use iter_phase_windows() with data_df= directly.

    Example
    -------
        df = phase_windows_dataframe(window_size=50, overlap=0.25)
        print(df.groupby("segment")["episode_index"].count())
    """
    skip_keys = {"frames", "motor_data"}
    rows = [
        {k: v for k, v in sample.items() if k not in skip_keys}
        for sample in iter_phase_windows(phases_csv, window_size, overlap, segments)
    ]
    return pd.DataFrame(rows)


# ── quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    window_size = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    overlap     = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

    print(f"window_size={window_size}  overlap={overlap}\n")

    # count total windows per segment
    counts = {"approach": 0, "carry": 0, "full": 0}
    total  = 0
    for sample in iter_phase_windows(window_size=window_size, overlap=overlap):
        counts[sample["segment"]] += 1
        total += 1

    print("Windows per segment:")
    for seg, n in counts.items():
        print(f"  {seg:10s}  {n:6d}")
    print(f"  {'TOTAL':10s}  {total:6d}")

    # show first 3 samples
    print("\nFirst 3 samples:")
    for i, sample in enumerate(iter_phase_windows(window_size=window_size, overlap=overlap)):
        print(f"  ep={sample['episode_index']:2d}  seg={sample['segment']:10s}  "
              f"frames={sample['frames'][0]}..{sample['frames'][-1]}  "
              f"pickup={sample['pickup_frame']}  drop={sample['drop_frame']}")
        if i >= 2:
            break
