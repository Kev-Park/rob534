"""
Build a per-episode and per-model summary table for the three eval datasets.

For each episode this computes:
  - Outcome label (from Excel)
  - Struggle-score S stats: min, max, mean, median, std  (5-channel normalised metric)
  - Gripper-based execution times: pickup_time_s, drop_time_s, hold_time_s
    (detected via gripper threshold crossing, same logic as batch_detect_phases.py)

Outputs
-------
  outputs/model_eval_episodes.csv   — one row per episode
  outputs/model_eval_summary.csv    — one row per model (success rate + S stats +
                                      execution time stats)

Usage
-----
    python make_model_eval_table.py
    python make_model_eval_table.py --excel "C:/path/to/labels.xlsx" --out outputs/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

# --siblings in this repo ────────────────────────────────────────────────────
from stability_monitor.calibration import Thresholds
from stability_monitor.config import Config
from struggle_monitor import compute_struggle_score_series
from batch_detect_phases import detect_phases, FPS, GRIPPER_COL_IDX, DEFAULT_THRESHOLD


# --dataset / label config ───────────────────────────────────────────────────

DATASETS = [
    {
        "id":          "nc8304/eval_smolvla-phase-split_combined",
        "short_name":  "phase-split combined",
        "excel_sheet": "nc8304eval_smolvla-phase-split_",
        "label_col":   "Ground truth outcome label",
    },
    {
        "id":          "nc8304/eval_smolvla-aug",
        "short_name":  "aug",
        "excel_sheet": "nc8304eval_smolvla-aug",
        "label_col":   "Label",
    },
    {
        "id":          "nc8304/eval_smolvla-phase-split-new-prompts",
        "short_name":  "phase-split new-prompts",
        "excel_sheet": "nc8304eval_smolvla-phase-split-",
        "label_col":   "Label",
    },
]

DEFAULT_EXCEL = Path(r"C:\Users\calle\Downloads\Outcome labels for model eval.xlsx")
DEFAULT_OUT   = Path(__file__).parent / "outputs"
THRESHOLDS_PATH = Path(__file__).parent / "thresholds" / "pooled.json"


# --helpers ──────────────────────────────────────────────────────────────────

def _load_labels(excel_path: Path, sheet: str, label_col: str) -> dict[int, str]:
    """Return {episode_index: label_lower} from one Excel sheet."""
    df = pd.read_excel(excel_path, sheet_name=sheet)
    out = {}
    for _, row in df.iterrows():
        ep = int(row["Episode Number"])
        lbl = row.get(label_col, None)
        if pd.notna(lbl):
            out[ep] = str(lbl).strip().lower()
    return out


def _load_parquet(dataset_id: str) -> pd.DataFrame:
    """Download (or use cache) and load all parquet chunks into one DataFrame."""
    print(f"  Loading {dataset_id} …", flush=True)
    repo_dir = Path(snapshot_download(repo_id=dataset_id, repo_type="dataset"))
    parquets = sorted((repo_dir / "data").rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files under {repo_dir}/data/")
    return pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)


def _episode_row(
    dataset_id: str,
    short_name: str,
    ep_idx: int,
    label: str | None,
    actions: np.ndarray,
    states: np.ndarray,
    thresholds: Thresholds,
    cfg: Config,
) -> dict:
    """Compute all metrics for one episode and return as a flat dict."""
    row: dict = {
        "model":         short_name,
        "episode_index": ep_idx,
        "label":         label or "unknown",
        "success":       (label or "").lower() == "success",
    }

    # --Struggle score S ─────────────────────────────────────────────────────
    try:
        S_series = compute_struggle_score_series(actions, states, thresholds, cfg)
        valid = S_series[~np.isnan(S_series)]
        if valid.size > 0:
            row["S_min"]    = round(float(valid.min()),              4)
            row["S_max"]    = round(float(valid.max()),              4)
            row["S_mean"]   = round(float(valid.mean()),             4)
            row["S_median"] = round(float(np.median(valid)),         4)
            row["S_std"]    = round(float(valid.std()),              4)
            row["S_p95"]    = round(float(np.percentile(valid, 95)), 4)
        else:
            for k in ("S_min", "S_max", "S_mean", "S_median", "S_std", "S_p95"):
                row[k] = float("nan")
    except Exception as exc:
        print(f"    [warn] S computation failed ep {ep_idx}: {exc}")
        for k in ("S_min", "S_max", "S_mean", "S_median", "S_std", "S_p95"):
            row[k] = float("nan")

    # --Gripper-based execution times ────────────────────────────────────────
    gripper = states[:, GRIPPER_COL_IDX]
    pickup_frame, drop_frame = detect_phases(gripper, DEFAULT_THRESHOLD)

    row["pickup_time_s"] = round(pickup_frame / FPS, 3) if pickup_frame is not None else None
    row["drop_time_s"]   = round(drop_frame   / FPS, 3) if drop_frame   is not None else None

    if pickup_frame is not None and drop_frame is not None:
        row["hold_time_s"] = round((drop_frame - pickup_frame) / FPS, 3)
    else:
        row["hold_time_s"] = None

    row["n_frames"]      = len(actions)
    row["episode_dur_s"] = round(len(actions) / FPS, 1)

    return row


# --main ─────────────────────────────────────────────────────────────────────

def main(excel_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = Thresholds.load(THRESHOLDS_PATH)
    cfg        = Config()

    all_rows: list[dict] = []

    for ds_cfg in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset : {ds_cfg['id']}")
        print(f"Model   : {ds_cfg['short_name']}")

        # labels from Excel
        labels = _load_labels(excel_path, ds_cfg["excel_sheet"], ds_cfg["label_col"])
        print(f"  Labels loaded: {len(labels)} episodes")

        # raw parquet data
        df_all = _load_parquet(ds_cfg["id"])

        for ep_idx in sorted(df_all["episode_index"].unique()):
            ep_idx = int(ep_idx)
            ep_df  = df_all[df_all["episode_index"] == ep_idx].sort_values("frame_index")
            actions = np.stack(ep_df["action"].values).astype(np.float64)
            states  = np.stack(ep_df["observation.state"].values).astype(np.float64)
            label   = labels.get(ep_idx)

            row = _episode_row(
                ds_cfg["id"], ds_cfg["short_name"],
                ep_idx, label, actions, states, thresholds, cfg,
            )
            all_rows.append(row)

        n_labeled = sum(1 for r in all_rows if r["model"] == ds_cfg["short_name"] and r["label"] != "unknown")
        print(f"  Episodes processed: {len([r for r in all_rows if r['model'] == ds_cfg['short_name']])}")
        print(f"  Labeled: {n_labeled}")

    ep_df = pd.DataFrame(all_rows)
    ep_csv = out_dir / "model_eval_episodes.csv"
    ep_df.to_csv(ep_csv, index=False)
    print(f"\nPer-episode table -> {ep_csv}")

    # --summary table ────────────────────────────────────────────────────────
    summary_rows = []
    for ds_cfg in DATASETS:
        m       = ds_cfg["short_name"]
        sub     = ep_df[ep_df["model"] == m]
        labeled = sub[sub["label"] != "unknown"]
        success = labeled[labeled["success"]]
        failure = labeled[~labeled["success"]]

        n_total      = len(sub)
        n_labeled    = len(labeled)
        n_success    = len(success)
        success_rate = n_success / n_labeled if n_labeled > 0 else float("nan")

        gripper_det_rate = round(sub["pickup_time_s"].notna().mean(), 3)

        def _s(df_slice, col, fn):
            vals = df_slice[col].dropna()
            return round(float(fn(vals)), 4) if len(vals) > 0 else float("nan")

        def _t(df_slice, col, fn):
            vals = df_slice[col].dropna()
            return round(float(fn(vals)), 3) if len(vals) > 0 else float("nan")

        summary_rows.append({
            "model":                  m,
            "n_episodes":             n_total,
            "n_labeled":              n_labeled,
            "n_success":              n_success,
            "success_rate":           round(success_rate, 3),
            "gripper_det_rate":       gripper_det_rate,
            # S stats across all episodes
            "S_min":                  _s(sub, "S_min",    np.min),
            "S_max":                  _s(sub, "S_max",    np.max),
            "S_mean":                 _s(sub, "S_mean",   np.mean),
            "S_std":                  _s(sub, "S_mean",   np.std),
            "S_median":               _s(sub, "S_median", np.median),
            "S_p95":                  _s(sub, "S_mean",   lambda x: np.percentile(x, 95)),
            # S mean split by outcome
            "S_mean_success":         _s(success, "S_mean", np.mean),
            "S_mean_failure":         _s(failure, "S_mean", np.mean),
            # execution times (all episodes with detected pickup)
            "pickup_time_mean_s":     _t(sub, "pickup_time_s", np.mean),
            "pickup_time_std_s":      _t(sub, "pickup_time_s", np.std),
            "pickup_time_min_s":      _t(sub, "pickup_time_s", np.min),
            "pickup_time_max_s":      _t(sub, "pickup_time_s", np.max),
            "drop_time_mean_s":       _t(sub, "drop_time_s",   np.mean),
            "drop_time_std_s":        _t(sub, "drop_time_s",   np.std),
            "hold_time_mean_s":       _t(sub, "hold_time_s",   np.mean),
            "hold_time_std_s":        _t(sub, "hold_time_s",   np.std),
            "hold_time_min_s":        _t(sub, "hold_time_s",   np.min),
            "hold_time_max_s":        _t(sub, "hold_time_s",   np.max),
        })

    sum_df = pd.DataFrame(summary_rows)
    sum_csv = out_dir / "model_eval_summary.csv"
    sum_df.to_csv(sum_csv, index=False)
    print(f"Summary table       -> {sum_csv}")

    # Print a human-readable version
    print("\n" + "="*90)
    print("MODEL SUMMARY")
    print("="*90)

    print("\n--Outcome --")
    print(sum_df[["model", "n_labeled", "n_success", "success_rate"]].to_string(index=False))

    print("\n--Stability Score S (per-episode mean, aggregated across episodes) --")
    print(sum_df[["model", "S_min", "S_max", "S_mean", "S_std", "S_median", "S_p95",
                  "S_mean_success", "S_mean_failure"]].to_string(index=False))

    print("\n--Execution Times (gripper-detected; pickup=grasp, drop=release, hold=pickup->drop) --")
    print(sum_df[["model", "gripper_det_rate",
                  "pickup_time_mean_s", "pickup_time_std_s", "pickup_time_min_s", "pickup_time_max_s",
                  "drop_time_mean_s", "drop_time_std_s",
                  "hold_time_mean_s", "hold_time_std_s", "hold_time_min_s", "hold_time_max_s",
                  ]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--excel", type=Path, default=DEFAULT_EXCEL,
        help="Path to 'Outcome labels for model eval.xlsx'",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help="Output directory",
    )
    args = parser.parse_args()

    if not args.excel.exists():
        print(f"ERROR: Excel file not found: {args.excel}", file=sys.stderr)
        sys.exit(1)

    main(args.excel, args.out)
