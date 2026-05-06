"""
Build a per-episode and per-model summary table for the three eval datasets.

For each episode this computes:
  - Outcome label (from Excel) + success flag
  - Gripper-based execution times: pickup_time_s, drop_time_s, hold_time_s
  - Struggle-score S stats: mean, max, min  (0-1, mean of 5 normalised channels)
  - Per-channel normalised metric stats (mean + max over episode windows):
      E_RMS, J_RMS, neg_SPARC, rho_HF, sigma_bar

Outputs
-------
  outputs/model_eval_episodes.csv    — one row per episode (all metrics)
  outputs/model_eval_summary.csv     — one row per model
  outputs/model_eval_table.xlsx      — formatted Excel: one sheet per model,
                                       success rows highlighted green,
                                       plus a summary sheet

Usage
-----
    python make_model_eval_table.py
    python make_model_eval_table.py --excel "C:/path/to/labels.xlsx" --out outputs/
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from stability_monitor.calibration import Thresholds, _per_channel_batch, M_CHANNELS
from stability_monitor.config import Config
from struggle_monitor import compute_struggle_score_series
from batch_detect_phases import detect_phases, FPS, GRIPPER_COL_IDX, DEFAULT_THRESHOLD

# ── config ───────────────────────────────────────────────────────────────────

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

DEFAULT_EXCEL   = Path(r"C:\Users\calle\Downloads\Outcome labels for model eval.xlsx")
DEFAULT_OUT     = Path(__file__).parent / "outputs"
THRESHOLDS_PATH = Path(__file__).parent / "thresholds" / "pooled.json"

# Column display names for each channel
CH_LABELS = {
    "E_RMS":     "E_RMS (track err)",
    "J_RMS":     "J_RMS (jerk)",
    "neg_SPARC": "neg_SPARC (smooth)",
    "rho_HF":    "rho_HF (HF pwr)",
    "sigma_bar": "sigma_bar (stall)",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_labels(excel_path: Path, sheet: str, label_col: str) -> dict[int, str]:
    df = pd.read_excel(excel_path, sheet_name=sheet)
    out = {}
    for _, row in df.iterrows():
        ep  = int(row["Episode Number"])
        lbl = row.get(label_col, None)
        if pd.notna(lbl):
            out[ep] = str(lbl).strip().lower()
    return out


def _load_parquet(dataset_id: str) -> pd.DataFrame:
    print(f"  Loading {dataset_id} ...", flush=True)
    repo_dir = Path(snapshot_download(repo_id=dataset_id, repo_type="dataset"))
    parquets = sorted((repo_dir / "data").rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files under {repo_dir}/data/")
    return pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)


def _channel_norm_series(
    actions: np.ndarray,
    states: np.ndarray,
    thresholds: Thresholds,
    cfg: Config,
) -> dict[str, np.ndarray]:
    """Return per-channel normalised arrays (clipped [0,1]), same logic as
    compute_struggle_score_series but per-channel so we can report them
    individually."""
    ch_arrays = _per_channel_batch(actions, states, cfg, thresholds)
    out = {}
    for i, ch_name in enumerate(M_CHANNELS):
        arr     = np.asarray(ch_arrays[ch_name], dtype=np.float64)
        theta_i = thresholds.theta[i]
        if ch_name == "sigma_bar":
            out[ch_name] = arr          # raw rate, already [0,1]
        elif theta_i > 0:
            out[ch_name] = np.clip(arr / theta_i, 0.0, 1.0)
        else:
            out[ch_name] = np.zeros_like(arr)
    return out


def _episode_row(
    short_name: str,
    ep_idx: int,
    label: str | None,
    actions: np.ndarray,
    states: np.ndarray,
    thresholds: Thresholds,
    cfg: Config,
) -> dict:
    row: dict = {
        "model":         short_name,
        "episode_index": ep_idx,
        "label":         label or "unknown",
        "success":       (label or "").lower() == "success",
    }

    # ── S score ───────────────────────────────────────────────────────────────
    try:
        S_series = compute_struggle_score_series(actions, states, thresholds, cfg)
        valid    = S_series[~np.isnan(S_series)]
        if valid.size > 0:
            row["S_mean"] = round(float(valid.mean()), 4)
            row["S_max"]  = round(float(valid.max()),  4)
            row["S_min"]  = round(float(valid.min()),  4)
            row["S_std"]  = round(float(valid.std()),  4)
        else:
            row["S_mean"] = row["S_max"] = row["S_min"] = row["S_std"] = float("nan")
    except Exception as exc:
        print(f"    [warn] S failed ep {ep_idx}: {exc}")
        row["S_mean"] = row["S_max"] = row["S_min"] = row["S_std"] = float("nan")

    # ── per-channel normalised stats ──────────────────────────────────────────
    try:
        ch_series = _channel_norm_series(actions, states, thresholds, cfg)
        for ch_name in M_CHANNELS:
            arr   = ch_series[ch_name]
            valid = arr[~np.isnan(arr)]
            if valid.size > 0:
                row[f"{ch_name}_mean"] = round(float(valid.mean()), 4)
                row[f"{ch_name}_max"]  = round(float(valid.max()),  4)
            else:
                row[f"{ch_name}_mean"] = row[f"{ch_name}_max"] = float("nan")
    except Exception as exc:
        print(f"    [warn] channel stats failed ep {ep_idx}: {exc}")
        for ch_name in M_CHANNELS:
            row[f"{ch_name}_mean"] = row[f"{ch_name}_max"] = float("nan")

    # ── gripper execution times ───────────────────────────────────────────────
    gripper    = states[:, GRIPPER_COL_IDX]
    pf, df_    = detect_phases(gripper, DEFAULT_THRESHOLD)
    row["pickup_time_s"] = round(pf  / FPS, 2) if pf  is not None else None
    row["drop_time_s"]   = round(df_ / FPS, 2) if df_ is not None else None
    row["hold_time_s"]   = (round((df_ - pf) / FPS, 2)
                            if pf is not None and df_ is not None else None)
    row["episode_dur_s"] = round(len(actions) / FPS, 1)

    return row


# ── Excel writer ──────────────────────────────────────────────────────────────

def _write_excel(ep_df: pd.DataFrame, sum_df: pd.DataFrame, path: Path) -> None:
    """Write a formatted .xlsx with one sheet per model + a summary sheet."""
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        wb = writer.book

        # shared formats
        hdr_fmt   = wb.add_format({"bold": True, "bg_color": "#2C3E50",
                                   "font_color": "white", "border": 1,
                                   "text_wrap": True, "valign": "vcenter",
                                   "align": "center"})
        ok_fmt    = wb.add_format({"bg_color": "#D5F5E3", "border": 1})
        fail_fmt  = wb.add_format({"bg_color": "#FDEDEC", "border": 1})
        num_ok    = wb.add_format({"bg_color": "#D5F5E3", "border": 1,
                                   "num_format": "0.00"})
        num_fail  = wb.add_format({"bg_color": "#FDEDEC", "border": 1,
                                   "num_format": "0.00"})
        pct_ok    = wb.add_format({"bg_color": "#D5F5E3", "border": 1,
                                   "num_format": "0%"})
        pct_fail  = wb.add_format({"bg_color": "#FDEDEC", "border": 1,
                                   "num_format": "0%"})
        bold_fmt  = wb.add_format({"bold": True, "border": 1,
                                   "bg_color": "#EBF5FB"})
        num_bold  = wb.add_format({"bold": True, "border": 1,
                                   "bg_color": "#EBF5FB", "num_format": "0.00"})
        pct_bold  = wb.add_format({"bold": True, "border": 1,
                                   "bg_color": "#EBF5FB", "num_format": "0%"})

        # ── column spec for per-episode sheets ────────────────────────────────
        ep_cols = [
            # (header, df_col, width, is_pct)
            ("Episode",         "episode_index", 8,  False),
            ("Outcome",         "label",         28, False),
            ("Success",         "success",        8, False),
            ("Pickup (s)",      "pickup_time_s", 10, False),
            ("Drop (s)",        "drop_time_s",   10, False),
            ("Hold (s)",        "hold_time_s",   10, False),
            ("Ep dur (s)",      "episode_dur_s", 10, False),
            ("S mean",          "S_mean",        10, False),
            ("S max",           "S_max",         10, False),
            ("S min",           "S_min",         10, False),
            ("S std",           "S_std",         10, False),
        ]
        for ch in M_CHANNELS:
            ep_cols.append((f"{CH_LABELS[ch]}\nmean", f"{ch}_mean", 14, False))
            ep_cols.append((f"{CH_LABELS[ch]}\nmax",  f"{ch}_max",  14, False))

        for ds_cfg in DATASETS:
            model     = ds_cfg["short_name"]
            sheet_sub = ep_df[ep_df["model"] == model].copy()
            sheet_sub = sheet_sub.sort_values("episode_index").reset_index(drop=True)

            # Add worksheet manually (no to_excel pre-fill — avoids stray columns)
            ws = wb.add_worksheet(model)
            writer.sheets[model] = ws

            ws.freeze_panes(1, 0)
            ws.set_row(0, 36)

            # write headers
            for col_i, (hdr, _, width, _) in enumerate(ep_cols):
                ws.write(0, col_i, hdr, hdr_fmt)
                ws.set_column(col_i, col_i, width)

            # write data rows
            for row_i, ep_row in sheet_sub.iterrows():
                success  = bool(ep_row["success"])
                txt_fmt  = ok_fmt   if success else fail_fmt
                num_fmt_ = num_ok   if success else num_fail
                pct_fmt_ = pct_ok   if success else pct_fail
                excel_r  = row_i + 1

                for col_i, (_, col, _, is_pct) in enumerate(ep_cols):
                    val = ep_row.get(col)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        ws.write_blank(excel_r, col_i, None, txt_fmt)
                    elif isinstance(val, bool):
                        ws.write_string(excel_r, col_i,
                                        "Yes" if val else "No", txt_fmt)
                    elif isinstance(val, (int, np.integer)):
                        ws.write_number(excel_r, col_i, int(val), txt_fmt)
                    elif isinstance(val, (float, np.floating)):
                        fmt = pct_fmt_ if is_pct else num_fmt_
                        ws.write_number(excel_r, col_i, float(val), fmt)
                    else:
                        ws.write_string(excel_r, col_i, str(val), txt_fmt)

            # summary average row at the bottom
            avg_r = len(sheet_sub) + 1
            ws.write_string(avg_r, 0, "AVG", bold_fmt)
            ws.write_string(avg_r, 1, f"{model}  ({len(sheet_sub)} eps)", bold_fmt)
            n_ok = int(sheet_sub["success"].sum())
            ws.write_string(avg_r, 2, f"{n_ok}/{len(sheet_sub)}", bold_fmt)
            for col_i, (_, col, _, is_pct) in enumerate(ep_cols):
                if col_i < 3:
                    continue
                vals = pd.to_numeric(sheet_sub[col], errors="coerce").dropna()
                if len(vals):
                    fmt = pct_bold if is_pct else num_bold
                    ws.write_number(avg_r, col_i, float(vals.mean()), fmt)
                else:
                    ws.write_blank(avg_r, col_i, None, bold_fmt)

        # ── summary sheet ─────────────────────────────────────────────────────
        sum_cols = [
            ("Model",             "model",                  24, False),
            ("Episodes",          "n_episodes",             10, False),
            ("Labeled",           "n_labeled",              10, False),
            ("Successes",         "n_success",              10, False),
            ("Success rate",      "success_rate",           12, True),
            ("S mean",            "S_mean",                 10, False),
            ("S max",             "S_max",                  10, False),
            ("S min",             "S_min",                  10, False),
            ("S std",             "S_std",                  10, False),
        ]
        for ch in M_CHANNELS:
            sum_cols.append((f"{CH_LABELS[ch]}\nmean", f"{ch}_mean_all", 14, False))
            sum_cols.append((f"{CH_LABELS[ch]}\nmax",  f"{ch}_max_all",  14, False))
        sum_cols += [
            ("Pickup mean (s)",   "pickup_time_mean_s",  14, False),
            ("Pickup std (s)",    "pickup_time_std_s",   12, False),
            ("Drop mean (s)",     "drop_time_mean_s",    12, False),
            ("Drop std (s)",      "drop_time_std_s",     12, False),
            ("Hold mean (s)",     "hold_time_mean_s",    12, False),
            ("Hold std (s)",      "hold_time_std_s",     12, False),
            ("Hold min (s)",      "hold_time_min_s",     12, False),
            ("Hold max (s)",      "hold_time_max_s",     12, False),
            ("Gripper det rate",  "gripper_det_rate",    14, True),
        ]

        # build the summary df columns we need
        for ch in M_CHANNELS:
            sum_df[f"{ch}_mean_all"] = ep_df.groupby("model")[f"{ch}_mean"].mean().reindex(sum_df["model"]).values
            sum_df[f"{ch}_max_all"]  = ep_df.groupby("model")[f"{ch}_max"].max().reindex(sum_df["model"]).values

        ws = wb.add_worksheet("Summary")
        writer.sheets["Summary"] = ws
        ws.freeze_panes(1, 0)
        ws.set_row(0, 36)
        num_fmt2 = wb.add_format({"num_format": "0.00"})
        pct_fmt2 = wb.add_format({"num_format": "0%"})
        for col_i, (hdr, _, width, _) in enumerate(sum_cols):
            ws.write(0, col_i, hdr, hdr_fmt)
            ws.set_column(col_i, col_i, width)
        for row_i, srow in sum_df.iterrows():
            for col_i, (_, col, _, is_pct) in enumerate(sum_cols):
                val = srow.get(col)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    ws.write_blank(row_i + 1, col_i, None)
                elif isinstance(val, (int, np.integer)):
                    ws.write_number(row_i + 1, col_i, int(val))
                elif isinstance(val, (float, np.floating)):
                    ws.write_number(row_i + 1, col_i, float(val),
                                    pct_fmt2 if is_pct else num_fmt2)
                else:
                    ws.write_string(row_i + 1, col_i, str(val))

    print(f"Excel table         -> {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main(excel_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = Thresholds.load(THRESHOLDS_PATH)
    cfg        = Config()

    all_rows: list[dict] = []

    for ds_cfg in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset : {ds_cfg['id']}")
        print(f"Model   : {ds_cfg['short_name']}")

        labels = _load_labels(excel_path, ds_cfg["excel_sheet"], ds_cfg["label_col"])
        print(f"  Labels: {len(labels)}")

        df_all = _load_parquet(ds_cfg["id"])

        for ep_idx in sorted(df_all["episode_index"].unique()):
            ep_idx  = int(ep_idx)
            ep_df_  = df_all[df_all["episode_index"] == ep_idx].sort_values("frame_index")
            actions = np.stack(ep_df_["action"].values).astype(np.float64)
            states  = np.stack(ep_df_["observation.state"].values).astype(np.float64)
            all_rows.append(_episode_row(
                ds_cfg["short_name"], ep_idx, labels.get(ep_idx),
                actions, states, thresholds, cfg,
            ))

        n = sum(1 for r in all_rows if r["model"] == ds_cfg["short_name"])
        print(f"  Processed: {n} episodes")

    ep_df = pd.DataFrame(all_rows)
    ep_df.to_csv(out_dir / "model_eval_episodes.csv", index=False)
    print(f"\nPer-episode CSV -> {out_dir / 'model_eval_episodes.csv'}")

    # ── summary ───────────────────────────────────────────────────────────────
    summary_rows = []
    for ds_cfg in DATASETS:
        m       = ds_cfg["short_name"]
        sub     = ep_df[ep_df["model"] == m]
        labeled = sub[sub["label"] != "unknown"]
        n_ok    = int(labeled["success"].sum())

        def _agg(df_, col, fn):
            v = df_[col].dropna()
            return round(float(fn(v)), 4) if len(v) else float("nan")

        r = {
            "model":              m,
            "n_episodes":         len(sub),
            "n_labeled":          len(labeled),
            "n_success":          n_ok,
            "success_rate":       round(n_ok / len(labeled), 3) if len(labeled) else float("nan"),
            "S_mean":             _agg(sub, "S_mean", np.mean),
            "S_max":              _agg(sub, "S_max",  np.max),
            "S_min":              _agg(sub, "S_min",  np.min),
            "S_std":              _agg(sub, "S_std",  np.mean),
            "pickup_time_mean_s": _agg(sub, "pickup_time_s", np.mean),
            "pickup_time_std_s":  _agg(sub, "pickup_time_s", np.std),
            "drop_time_mean_s":   _agg(sub, "drop_time_s",   np.mean),
            "drop_time_std_s":    _agg(sub, "drop_time_s",   np.std),
            "hold_time_mean_s":   _agg(sub, "hold_time_s",   np.mean),
            "hold_time_std_s":    _agg(sub, "hold_time_s",   np.std),
            "hold_time_min_s":    _agg(sub, "hold_time_s",   np.min),
            "hold_time_max_s":    _agg(sub, "hold_time_s",   np.max),
            "gripper_det_rate":   round(sub["pickup_time_s"].notna().mean(), 3),
        }
        for ch in M_CHANNELS:
            r[f"{ch}_mean"] = _agg(sub, f"{ch}_mean", np.mean)
            r[f"{ch}_max"]  = _agg(sub, f"{ch}_max",  np.max)
        summary_rows.append(r)

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(out_dir / "model_eval_summary.csv", index=False)
    print(f"Summary CSV      -> {out_dir / 'model_eval_summary.csv'}")

    _write_excel(ep_df, sum_df, out_dir / "model_eval_table.xlsx")

    # ── console print ─────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("OUTCOME")
    print(sum_df[["model","n_labeled","n_success","success_rate"]].to_string(index=False))
    print("\nSTABILITY  S (mean across episodes)")
    print(sum_df[["model","S_min","S_max","S_mean","S_std"]].to_string(index=False))
    ch_mean_cols = ["model"] + [f"{ch}_mean" for ch in M_CHANNELS]
    print("\nPER-CHANNEL means (normalised 0-1)")
    print(sum_df[ch_mean_cols].to_string(index=False))
    print("\nEXECUTION TIMES")
    print(sum_df[["model","pickup_time_mean_s","drop_time_mean_s",
                  "hold_time_mean_s","hold_time_std_s"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--out",   type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.excel.exists():
        print(f"ERROR: Excel not found: {args.excel}", file=sys.stderr)
        sys.exit(1)
    main(args.excel, args.out)
