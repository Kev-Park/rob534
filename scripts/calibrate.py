"""Calibrate stability-monitor thresholds from success-labelled eval episodes.

Default behaviour: load the ``success``-labelled episodes from each of the
three ``nc8304/eval_*`` datasets, fit thresholds per dataset, and *also*
fit a pooled set across all three. Each calibration writes a JSON file
(canonical, runtime-loadable via :class:`Thresholds.load`) and a CSV file
(long format, human-readable summary) under ``--out-dir``.

Output layout (``--out-dir thresholds`` by default)::

    thresholds/
      eval_smolvla-aug.json
      eval_smolvla-aug.csv
      eval_smolvla-phase-split-new-prompts.json
      eval_smolvla-phase-split-new-prompts.csv
      eval_smolvla-phase-split_combined.json
      eval_smolvla-phase-split_combined.csv
      pooled.json
      pooled.csv

Usage::

    python scripts/calibrate.py
    python scripts/calibrate.py --datasets nc8304/eval_smolvla-aug
    python scripts/calibrate.py --no-pool        # skip the pooled fit
    python scripts/calibrate.py --no-per-dataset # only emit the pooled fit
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow direct invocation (``python scripts/calibrate.py``) by ensuring the
# project root is on sys.path so the ``stability_monitor`` package imports.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stability_monitor.calibration import (  # noqa: E402
    M_CHANNELS,
    Thresholds,
    calibrate,
)
from stability_monitor.config import Config  # noqa: E402
from stability_monitor.io.lerobot_loader import (  # noqa: E402
    EpisodeArrays,
    iter_episodes,
)


DEFAULT_DATASETS = (
    "nc8304/eval_smolvla-aug",
    "nc8304/eval_smolvla-phase-split-new-prompts",
    "nc8304/eval_smolvla-phase-split_combined",
)


def _slug_for(dataset_id: str) -> str:
    """Compact filename stem from a HuggingFace dataset id."""
    return dataset_id.split("/")[-1]


def _write_csv(
    thresholds: Thresholds,
    summary: dict,
    cfg: Config,
    csv_path: Path,
    name: str,
    n_episodes: int,
) -> None:
    """Long-format CSV: ``section, parameter, channel, value``.

    Sections:
      - ``meta``      : the dataset name and episode count
      - ``threshold`` : fitted Thresholds (theta, delta_a, delta_q, a_min, a_max)
      - ``summary``   : per-channel pooled stats (n_finite, median, p95, p99, max)
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "parameter", "channel", "value"])

        writer.writerow(["meta", "name", "-", name])
        writer.writerow(["meta", "n_episodes", "-", str(n_episodes)])

        for i, ch in enumerate(M_CHANNELS):
            writer.writerow(["threshold", "theta", ch, f"{thresholds.theta[i]:.10g}"])
        for i, joint in enumerate(cfg.joint_names):
            writer.writerow(["threshold", "delta_a", joint, f"{thresholds.delta_a[i]:.10g}"])
        for i, joint in enumerate(cfg.joint_names):
            writer.writerow(["threshold", "delta_q", joint, f"{thresholds.delta_q[i]:.10g}"])
        for i, joint in enumerate(cfg.joint_names):
            writer.writerow(["threshold", "a_min", joint, f"{thresholds.a_min[i]:.10g}"])
        for i, joint in enumerate(cfg.joint_names):
            writer.writerow(["threshold", "a_max", joint, f"{thresholds.a_max[i]:.10g}"])

        for ch in M_CHANNELS:
            stats = summary[ch]
            for stat_name in ("n_finite", "median", "p95", "p99", "max"):
                writer.writerow(["summary", stat_name, ch, f"{stats[stat_name]:.10g}"])


def _print_summary(
    name: str, thresholds: Thresholds, summary: dict, cfg: Config, n_episodes: int
) -> None:
    print()
    print("=" * 70)
    print(f"CALIBRATION: {name}  ({n_episodes} successful episode{'s' if n_episodes != 1 else ''})")
    print("=" * 70)
    print(f"{'channel':<12} {'n':>8} {'median':>12} {'p95':>12} {'p99':>12} {'max':>12}")
    for ch in M_CHANNELS:
        s = summary[ch]
        print(
            f"{ch:<12} {s['n_finite']:>8d} {s['median']:>12.4g} "
            f"{s['p95']:>12.4g} {s['p99']:>12.4g} {s['max']:>12.4g}"
        )
    print()
    print("theta (95th percentile per channel):")
    for i, ch in enumerate(M_CHANNELS):
        print(f"  theta[{ch:<10}] = {thresholds.theta[i]:.4g}")
    print()
    print(f"per-joint stall thresholds (deg) — 95th percentile of M={cfg.M}-step diffs:")
    print(f"{'joint':<16} {'delta_a':>10} {'delta_q':>10} {'a_min':>10} {'a_max':>10}")
    for i, joint in enumerate(cfg.joint_names):
        print(
            f"{joint:<16} {thresholds.delta_a[i]:>10.3f} "
            f"{thresholds.delta_q[i]:>10.3f} {thresholds.a_min[i]:>10.2f} "
            f"{thresholds.a_max[i]:>10.2f}"
        )


def _calibrate_and_write(
    name: str,
    episodes: list[EpisodeArrays],
    cfg: Config,
    margin_frac: float,
    out_dir: Path,
) -> bool:
    """Run calibration on one corpus and write JSON + CSV. Returns success."""
    if not episodes:
        print(f"\n[skip] {name}: no successful episodes loaded")
        return False
    if len(episodes) == 1:
        print(
            f"\n[warn] {name}: only 1 successful episode — calibrated "
            f"thresholds will be very noisy"
        )

    pairs = [(ep.action, ep.state) for ep in episodes]
    thresholds, summary = calibrate(pairs, cfg, margin_frac=margin_frac)
    json_path = out_dir / f"{name}.json"
    csv_path = out_dir / f"{name}.csv"
    thresholds.save(json_path)
    _write_csv(thresholds, summary, cfg, csv_path, name, len(episodes))
    _print_summary(name, thresholds, summary, cfg, len(episodes))
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="HuggingFace dataset ids to calibrate from",
    )
    parser.add_argument(
        "--no-successful-only",
        dest="successful_only",
        action="store_false",
        help="Disable success-label filtering",
    )
    parser.add_argument(
        "--no-per-dataset",
        dest="per_dataset",
        action="store_false",
        help="Skip emitting one Thresholds per dataset",
    )
    parser.add_argument(
        "--no-pool",
        dest="pool",
        action="store_false",
        help="Skip emitting the pooled (cross-dataset) Thresholds",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("episode_videos"),
        help="Directory holding per-dataset 'Outcome labels for model eval - <slug>.csv' files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("thresholds"),
        help="Directory for thresholds JSON + CSV outputs",
    )
    parser.add_argument(
        "--margin-frac",
        type=float,
        default=0.05,
        help="Symmetric margin added to action range for a_min/a_max",
    )
    args = parser.parse_args(argv)

    if not args.per_dataset and not args.pool:
        parser.error("--no-per-dataset and --no-pool together produce nothing to write")

    cfg = Config()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading episodes from {len(args.datasets)} dataset(s)"
        f"{' (success-only)' if args.successful_only else ''}..."
    )
    per_dataset: dict[str, list[EpisodeArrays]] = {}
    for ds in args.datasets:
        slug = _slug_for(ds)
        eps = list(
            iter_episodes(
                ds,
                only_successful=args.successful_only,
                labels_root=args.labels_root,
            )
        )
        per_dataset[slug] = eps
        print(f"  {slug:<60s} {len(eps):>4d} success-labelled episodes")

    n_written = 0
    if args.per_dataset:
        for slug, eps in per_dataset.items():
            if _calibrate_and_write(slug, eps, cfg, args.margin_frac, args.out_dir):
                n_written += 1

    if args.pool:
        all_eps: list[EpisodeArrays] = [e for eps in per_dataset.values() for e in eps]
        if _calibrate_and_write("pooled", all_eps, cfg, args.margin_frac, args.out_dir):
            n_written += 1

    if n_written == 0:
        print("\nERROR: no Thresholds files were written")
        return 1
    print(f"\nDone — {n_written} Thresholds set(s) written under {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
