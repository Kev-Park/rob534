"""Load LeRobot eval datasets, optionally filtered to success-labelled episodes.

The three eval datasets (``nc8304/eval_smolvla-aug``, ``-phase-split-new-prompts``,
``-phase-split_combined``) are pulled via :func:`huggingface_hub.snapshot_download`
and read directly from the flattened ``data/chunk-000/file-000.parquet`` (six-DOF
``action`` / ``observation.state`` at 30 Hz). The companion success/failure
labels live next to the per-episode mp4 clips at
``episode_videos/<dataset_slug>/Outcome labels for model eval - <slug>.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download


SUCCESS_LABEL = "success"
DEFAULT_LABELS_ROOT = Path("episode_videos")
# Accepted column names for the outcome label across the three label CSVs.
_LABEL_COLUMN_CANDIDATES = ("Ground truth outcome label", "Label")


@dataclass
class EpisodeArrays:
    """Per-episode arrays for the stability monitor.

    Attributes
    ----------
    dataset_id
        HuggingFace dataset id, e.g. ``nc8304/eval_smolvla-aug``.
    episode_index
        Episode index in the dataset's flat parquet file.
    label
        Outcome label from the success-labels CSV (``"success"``,
        ``"failure to grasp"``, etc.). ``None`` if no label was supplied.
    action, state, timestamp
        Stacked arrays of shape ``(T, 6)``, ``(T, 6)``, ``(T,)``.
    """

    dataset_id: str
    episode_index: int
    label: str | None
    action: np.ndarray
    state: np.ndarray
    timestamp: np.ndarray


def labels_csv_for(
    dataset_id: str, root: Path | str = DEFAULT_LABELS_ROOT
) -> Path:
    """Resolve the labels CSV path for a HuggingFace dataset id.

    The split-into-mp4s script puts each dataset's clips and labels CSV
    under ``<root>/<slug>/`` where ``slug = dataset_id.replace("/", "__")``.
    The CSV filename uses single underscores between namespace and name.
    """
    slug_double = dataset_id.replace("/", "__")
    slug_single = dataset_id.replace("/", "_")
    return Path(root) / slug_double / (
        f"Outcome labels for model eval - {slug_single}.csv"
    )


def _load_labels(labels_csv: Path) -> dict[int, str]:
    """Return ``{episode_index: label_lowercased}`` from a labels CSV.

    Accepts either of the two schemas used across the three eval datasets:
    ``Episode Number, Ground truth outcome label`` (lowercase free-text
    labels) or ``Episode Number, Label`` (capitalised ``Success`` /
    ``Failure``). Labels are normalised to lowercase for comparison.
    """
    df = pd.read_csv(labels_csv)
    if "Episode Number" not in df.columns:
        raise ValueError(
            f"missing 'Episode Number' column in {labels_csv}: "
            f"{df.columns.tolist()}"
        )
    label_col = next(
        (c for c in _LABEL_COLUMN_CANDIDATES if c in df.columns), None
    )
    if label_col is None:
        raise ValueError(
            f"no recognised label column in {labels_csv}: "
            f"{df.columns.tolist()} (expected one of {_LABEL_COLUMN_CANDIDATES})"
        )
    return {
        int(r["Episode Number"]): str(r[label_col]).strip().lower()
        for _, r in df.iterrows()
    }


def iter_episodes(
    dataset_id: str,
    only_successful: bool = False,
    labels_csv: Path | str | None = None,
    labels_root: Path | str = DEFAULT_LABELS_ROOT,
) -> Iterator[EpisodeArrays]:
    """Stream episodes from a HuggingFace eval dataset.

    Parameters
    ----------
    dataset_id
        HuggingFace dataset id.
    only_successful
        If ``True``, yield only episodes whose outcome label is ``"success"``.
    labels_csv
        Explicit path to the labels CSV. If ``None`` the default
        ``<labels_root>/<slug>/Outcome labels for model eval - <slug>.csv``
        is used (only when filtering by success).
    labels_root
        Base directory for the per-dataset labels CSVs.
    """
    repo_dir = Path(snapshot_download(repo_id=dataset_id, repo_type="dataset"))

    labels: dict[int, str] = {}
    if only_successful or labels_csv is not None:
        path = (
            Path(labels_csv)
            if labels_csv is not None
            else labels_csv_for(dataset_id, labels_root)
        )
        if not path.exists():
            raise FileNotFoundError(
                f"labels CSV not found at {path}; pass --labels-csv or set "
                f"--labels-root"
            )
        labels = _load_labels(path)

    parquet_paths = sorted((repo_dir / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no data parquet files under {repo_dir}/data/")
    df = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)

    for ep_idx in sorted(df["episode_index"].unique()):
        ep_idx = int(ep_idx)
        label = labels.get(ep_idx)
        if only_successful and label != SUCCESS_LABEL:
            continue
        ep = df[df["episode_index"] == ep_idx].sort_values("frame_index")
        action = np.stack(ep["action"].values).astype(np.float64)
        state = np.stack(ep["observation.state"].values).astype(np.float64)
        timestamp = ep["timestamp"].to_numpy(dtype=np.float64)
        yield EpisodeArrays(
            dataset_id=dataset_id,
            episode_index=ep_idx,
            label=label,
            action=action,
            state=state,
            timestamp=timestamp,
        )
