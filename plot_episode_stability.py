"""
Save per-episode stability plots and a grid overview per model.

Per-episode plots  (outputs/plots/<slug>/ep_NNN_<label>.png)
  Row 1 : S(t) with P95 threshold and pickup/drop markers
  Rows 2-6 : raw channel values (E_RMS, J_RMS, neg_SPARC, rho_HF, sigma_bar)

Grid overview  (outputs/plots/<slug>_grid.png)
  Rows = episodes (sorted by index), Columns = metrics (S + 5 channels)
  Success episodes: bold green line + green spine
  Failure episodes: thin grey line
  Green / red tick marks show pickup / drop time in every cell
  Y-scale is shared within each column so episodes are directly comparable

Usage
-----
    python plot_episode_stability.py
    python plot_episode_stability.py --grid-only   # skip per-episode plots
    python plot_episode_stability.py --out outputs/plots
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from huggingface_hub import snapshot_download

from stability_monitor.calibration import Thresholds, _per_channel_batch, M_CHANNELS
from stability_monitor.config import Config
from struggle_monitor import compute_struggle_score_series
from batch_detect_phases import FPS

THRESHOLDS_PATH = Path(__file__).parent / "thresholds" / "pooled.json"
EPISODES_CSV    = Path(__file__).parent / "outputs" / "model_eval_episodes.csv"

DATASETS = [
    {"id": "nc8304/eval_smolvla-phase-split_combined",    "short_name": "phase-split combined",    "slug": "phase_split_combined"},
    {"id": "nc8304/eval_smolvla-aug",                     "short_name": "aug",                     "slug": "aug"},
    {"id": "nc8304/eval_smolvla-phase-split-new-prompts", "short_name": "phase-split new-prompts", "slug": "phase_split_new_prompts"},
]

# Column definitions for the grid: (key, display_label, unit_hint)
GRID_COLS = [
    ("S",         "S",         "normalised"),
    ("E_RMS",     "E_RMS",     "tracking err"),
    ("J_RMS",     "J_RMS",     "jerk"),
    ("neg_SPARC", "neg_SPARC", "smoothness"),
    ("rho_HF",    "rho_HF",    "HF power"),
    ("sigma_bar", "sigma_bar", "stall/clip"),
]

CHANNEL_LABELS = {
    "E_RMS":     "Tracking Error",
    "J_RMS":     "Jerk",
    "neg_SPARC": "neg_SPARC",
    "rho_HF":    "HF Power",
    "sigma_bar": "Stall/Clip",
}
CHANNEL_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

SUCCESS_COLOR  = "#1B5E20"
FAILURE_COLOR  = "#9E9E9E"
PICKUP_COLOR   = "#43A047"
DROP_COLOR     = "#E53935"


# ── helpers ───────────────────────────────────────────────────────────────────

def _compute_arrays(actions, states, thresholds, cfg):
    """Return S_series (T,) and {ch_name: array (T,)} of raw channel values.

    S_series uses compute_struggle_score_series() — the same normalization
    as the live rollout monitor — so offline analysis matches live behaviour.
    """
    S_series = compute_struggle_score_series(actions, states, thresholds, cfg)
    ch = _per_channel_batch(actions, states, cfg, thresholds)
    raw = {name: np.asarray(ch[name], dtype=np.float64) for name in M_CHANNELS}
    return S_series, raw


def _label_abbrev(label: str) -> str:
    """Short label string for tight row annotations."""
    label = label.lower()
    if "success" in label:
        return "ok"
    if "timeout" in label:
        return "timeout"
    if "grasp" in label:
        return "fail:grasp"
    if "drop" in label:
        return "fail:drop"
    if "region" in label:
        return "fail:region"
    if "failure" in label:
        return "fail"
    return label[:10]


# ── per-episode plots ─────────────────────────────────────────────────────────

def save_episode_plots(ep_meta, df_all, short_name, slug, thresholds, cfg, out_dir):
    """Save one detailed PNG per episode.

    Returns (all_S, all_raw) where:
      all_S   : {ep_idx: S_series array}
      all_raw : {ch_name: {ep_idx: array}}
    """
    ep_dir = out_dir / slug
    ep_dir.mkdir(parents=True, exist_ok=True)

    all_S:   dict[int, np.ndarray] = {}
    all_raw: dict[str, dict[int, np.ndarray]] = {ch: {} for ch in M_CHANNELS}

    for ep_idx in sorted(df_all["episode_index"].unique()):
        ep_idx  = int(ep_idx)
        ep_df   = df_all[df_all["episode_index"] == ep_idx].sort_values("frame_index")
        actions = np.stack(ep_df["action"].values).astype(np.float64)
        states  = np.stack(ep_df["observation.state"].values).astype(np.float64)
        time_s  = np.arange(len(actions)) / FPS

        meta_row = ep_meta[ep_meta["episode_index"] == ep_idx]
        if meta_row.empty:
            label, success, pickup_t, drop_t = "unknown", False, None, None
        else:
            r        = meta_row.iloc[0]
            label    = r["label"]
            success  = bool(r["success"])
            pickup_t = r["pickup_time_s"] if pd.notna(r["pickup_time_s"]) else None
            drop_t   = r["drop_time_s"]   if pd.notna(r["drop_time_s"])   else None

        try:
            S_series, raw = _compute_arrays(actions, states, thresholds, cfg)
        except Exception as exc:
            print(f"  [warn] ep {ep_idx} compute failed: {exc}")
            continue

        all_S[ep_idx] = S_series
        for ch_name, arr in raw.items():
            all_raw[ch_name][ep_idx] = arr

        # ---- figure ---------------------------------------------------------
        fig = plt.figure(figsize=(10, 9))
        gs  = gridspec.GridSpec(6, 1, hspace=0.45, figure=fig)
        ax_S   = fig.add_subplot(gs[0])
        ax_chs = [fig.add_subplot(gs[i + 1], sharex=ax_S) for i in range(5)]
        plt.setp(ax_S.get_xticklabels(), visible=False)
        for ax in ax_chs[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)

        color_line = SUCCESS_COLOR if success else "#B71C1C"
        title = (
            f"{short_name}  ep {ep_idx}  |  {label}  |  "
            f"pickup={pickup_t:.1f}s  drop={drop_t:.1f}s"
            if pickup_t is not None and drop_t is not None
            else f"{short_name}  ep {ep_idx}  |  {label}"
        )

        ax_S.plot(time_s, S_series, color=color_line, linewidth=0.9, label="S(t)")
        ax_S.axhline(1.0, color="gray", linewidth=0.7, linestyle="--", alpha=0.8, label="P95")
        ax_S.fill_between(time_s, 0, S_series, where=S_series > 1.0,
                          alpha=0.15, color="red", label="S>1")
        ax_S.set_ylabel("S", fontsize=7)
        ax_S.set_ylim(bottom=0)
        ax_S.set_title(title, fontsize=7, pad=2)
        ax_S.tick_params(labelsize=6)
        ax_S.legend(fontsize=5, loc="upper right", framealpha=0.5)

        for ax, ch_name, color in zip(ax_chs, M_CHANNELS, CHANNEL_COLORS):
            ax.plot(time_s, raw[ch_name], color=color, linewidth=0.7)
            ax.set_ylabel(CHANNEL_LABELS[ch_name], fontsize=5.5)
            ax.tick_params(labelsize=5)
            ax.set_ylim(bottom=0)

        for ax in [ax_S] + ax_chs:
            if pickup_t is not None:
                ax.axvline(pickup_t, color=PICKUP_COLOR, linewidth=1.1)
            if drop_t is not None:
                ax.axvline(drop_t, color=DROP_COLOR, linewidth=1.1)

        ax_chs[-1].set_xlabel("time (s)", fontsize=6)

        label_slug = label.replace(" ", "_").replace("/", "-")[:30]
        fig.savefig(ep_dir / f"ep_{ep_idx:03d}_{label_slug}.png",
                    dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved {len(all_S)} episode plots -> {ep_dir}")
    return all_S, all_raw


# ── grid overview ─────────────────────────────────────────────────────────────

def save_grid_overview(all_S, all_raw, ep_meta, short_name, slug, out_dir):
    """One tall figure: rows = episodes, columns = metrics.

    Success rows have a bold green line and a green spine border.
    Failure rows have a thin grey line.
    Green / red vertical ticks show pickup / drop in every cell.
    Y-scale is shared within each column (P99 clipped).
    """
    ep_indices = sorted(all_S.keys())
    n_ep  = len(ep_indices)
    n_col = len(GRID_COLS)

    # ── per-column y limits (99th percentile across all episodes) ─────────────
    ylims: dict[str, float] = {}
    for key, *_ in GRID_COLS:
        if key == "S":
            vals = np.concatenate([all_S[i][~np.isnan(all_S[i])] for i in ep_indices])
        else:
            vals = np.concatenate([all_raw[key][i][~np.isnan(all_raw[key][i])]
                                   for i in ep_indices if i in all_raw[key]])
        ylims[key] = float(np.percentile(vals, 99)) if vals.size else 1.0

    # ── figure layout ─────────────────────────────────────────────────────────
    row_h  = 0.52            # inches per episode row
    col_w  = 2.1             # inches per metric column
    lbl_w  = 1.0             # inches for row label gutter
    fig_w  = lbl_w + n_col * col_w
    fig_h  = n_ep * row_h + 1.2   # +1.2 for column headers

    fig = plt.figure(figsize=(fig_w, fig_h))

    # GridSpec: one extra narrow column on the left for row labels
    gs = gridspec.GridSpec(
        n_ep, n_col + 1,
        figure=fig,
        hspace=0.04,
        wspace=0.06,
        left=0.01,
        right=0.99,
        top=0.97,
        bottom=0.02,
        width_ratios=[lbl_w / col_w] + [1.0] * n_col,
    )

    for row_i, ep_idx in enumerate(ep_indices):
        meta_row = ep_meta[ep_meta["episode_index"] == ep_idx]
        if meta_row.empty:
            label, success, pickup_t, drop_t = "unknown", False, None, None
        else:
            r        = meta_row.iloc[0]
            label    = r["label"]
            success  = bool(r["success"])
            pickup_t = r["pickup_time_s"] if pd.notna(r["pickup_time_s"]) else None
            drop_t   = r["drop_time_s"]   if pd.notna(r["drop_time_s"])   else None

        line_color = SUCCESS_COLOR if success else FAILURE_COLOR
        line_width = 1.3          if success else 0.55
        spine_color = SUCCESS_COLOR if success else "#BDBDBD"
        spine_width = 1.2          if success else 0.4

        S_arr = all_S[ep_idx]
        t_S   = np.arange(len(S_arr)) / FPS

        # ── row label cell (column 0) ─────────────────────────────────────────
        ax_lbl = fig.add_subplot(gs[row_i, 0])
        ax_lbl.set_axis_off()
        lbl_text = f"ep{ep_idx}  {_label_abbrev(label)}"
        ax_lbl.text(0.95, 0.5, lbl_text,
                    ha="right", va="center", fontsize=5.5,
                    fontweight="bold" if success else "normal",
                    color=SUCCESS_COLOR if success else "#444444",
                    transform=ax_lbl.transAxes)

        # ── metric cells ──────────────────────────────────────────────────────
        for col_i, (key, disp_label, unit) in enumerate(GRID_COLS):
            ax = fig.add_subplot(gs[row_i, col_i + 1])

            if key == "S":
                arr    = S_arr
                time_s = t_S
                ax.axhline(1.0, color="gray", linewidth=0.4,
                           linestyle="--", alpha=0.6, zorder=1)
                ax.fill_between(time_s, 0, arr, where=arr > 1.0,
                                alpha=0.12, color="red", zorder=2)
            else:
                arr    = all_raw[key].get(ep_idx, np.array([]))
                time_s = np.arange(len(arr)) / FPS

            ax.plot(time_s, arr, color=line_color, linewidth=line_width,
                    zorder=3, solid_capstyle="round")

            # pickup / drop ticks
            if pickup_t is not None:
                ax.axvline(pickup_t, color=PICKUP_COLOR,
                           linewidth=0.9 if success else 0.5, alpha=0.8, zorder=4)
            if drop_t is not None:
                ax.axvline(drop_t, color=DROP_COLOR,
                           linewidth=0.9 if success else 0.5, alpha=0.8, zorder=4)

            # axis limits and cleanup
            ax.set_ylim(0, ylims[key] * 1.05)
            if len(time_s):
                ax.set_xlim(0, time_s[-1])
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_linewidth(spine_width)
                spine.set_color(spine_color)

            # column headers on first row only
            if row_i == 0:
                ax.set_title(f"{disp_label}\n({unit})", fontsize=6.5,
                             fontweight="bold", pad=3)

    # ── legend strip at the very top ─────────────────────────────────────────
    fig.text(0.50, 0.992,
             f"{short_name}  —  {n_ep} episodes  "
             f"|  green border = success  |  green line = pickup  |  red line = drop",
             ha="center", va="top", fontsize=7.5, color="#333333")

    fname = out_dir / f"{slug}_grid.png"
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grid overview -> {fname}")


# ── main ─────────────────────────────────────────────────────────────────────

def main(out_dir: Path, grid_only: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = Thresholds.load(THRESHOLDS_PATH)
    cfg        = Config()
    ep_meta    = pd.read_csv(EPISODES_CSV)

    for ds_cfg in DATASETS:
        print(f"\n{'='*60}")
        print(f"Model: {ds_cfg['short_name']}")

        repo_dir = Path(snapshot_download(repo_id=ds_cfg["id"], repo_type="dataset"))
        parquets = sorted((repo_dir / "data").rglob("*.parquet"))
        df_all   = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
        meta     = ep_meta[ep_meta["model"] == ds_cfg["short_name"]].copy()

        if grid_only:
            # Compute arrays without saving per-episode plots
            print("  Computing metric arrays (grid-only mode)...")
            all_S:   dict[int, np.ndarray] = {}
            all_raw: dict[str, dict[int, np.ndarray]] = {ch: {} for ch in M_CHANNELS}
            for ep_idx in sorted(df_all["episode_index"].unique()):
                ep_idx  = int(ep_idx)
                ep_df   = df_all[df_all["episode_index"] == ep_idx].sort_values("frame_index")
                actions = np.stack(ep_df["action"].values).astype(np.float64)
                states  = np.stack(ep_df["observation.state"].values).astype(np.float64)
                try:
                    S_series, raw = _compute_arrays(actions, states, thresholds, cfg)
                    all_S[ep_idx] = S_series
                    for ch_name, arr in raw.items():
                        all_raw[ch_name][ep_idx] = arr
                except Exception as exc:
                    print(f"  [warn] ep {ep_idx}: {exc}")
        else:
            all_S, all_raw = save_episode_plots(
                meta, df_all,
                ds_cfg["short_name"], ds_cfg["slug"],
                thresholds, cfg, out_dir,
            )

        save_grid_overview(
            all_S, all_raw, meta,
            ds_cfg["short_name"], ds_cfg["slug"], out_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "outputs" / "plots")
    parser.add_argument("--grid-only", action="store_true",
                        help="Skip per-episode plots, only regenerate grid overviews")
    args = parser.parse_args()
    main(args.out, grid_only=args.grid_only)
