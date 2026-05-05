# Stability Monitor — Design Sketch

Source of truth for the math: `stability_monitoring_methods.tex`.
Source of truth for the API contract: `STABILITY_MONITOR_SPEC.md` (the prompt).

## Dataset binding

The "teacher corpus" for calibration is the union of episodes labelled
`success` in:

- `nc8304/eval_smolvla-aug`
- `nc8304/eval_smolvla-phase-split-new-prompts`
- `nc8304/eval_smolvla-phase-split_combined`

Each dataset has a sibling CSV at
`episode_videos/<dataset_slug>/Outcome labels for model eval - <slug>.csv`
with columns `Episode Number, Ground truth outcome label`.

Schema verified on `nc8304/eval_smolvla-phase-split_combined`, episode 0:

- `action`: `float32[T, 6]`
- `observation.state`: `float32[T, 6]`
- `timestamp`: `float64`, monotonic, `dt = 1/30 s` exactly
- 50 episodes; total ~46.8k frames
- No `observation.load` / `observation.current` columns. Optional torque
  extension is not available — the monitor will not advertise it.

Datasets are pulled with `huggingface_hub.snapshot_download` (cached on rerun)
and read directly from `data/chunk-*/file-*.parquet`. The spec mentions
`./data/so101_dataset` as the default; we override that default to the three
HF dataset IDs above.

## Config (`stability_monitor/config.py`)

```python
@dataclass(frozen=True)
class Config:
    fs:            float = 30.0
    N:             int   = 30      # window length in samples (~1 s)
    M:             int   = 5       # stall lookback in samples
    P:             int   = 3       # persistence filter length
    sg_window:     int   = 7
    sg_polyorder:  int   = 3
    omega_h:       float = 5.0     # HF crossover (Hz)
    omega_c_max:   float = 12.0    # SPARC cutoff cap (Hz)
    V_bar:         float = 0.05    # SPARC normalised-spectrum threshold
    eps_clip_frac: float = 0.02    # joint-clip tolerance fraction of range
    n_joints:      int   = 6
    joint_names: tuple[str, ...] = (
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex",   "wrist_roll",    "gripper",
    )
```

## Calibrated thresholds (`stability_monitor/calibration.py`)

```python
@dataclass(frozen=True)
class Thresholds:
    theta:    np.ndarray  # shape (5,)  95th-percentile of m_ell on success corpus
    delta_a:  np.ndarray  # shape (6,)  per-joint stall command threshold
    delta_q:  np.ndarray  # shape (6,)  per-joint stall motion threshold
    a_min:    np.ndarray  # shape (6,)  per-joint action range min (with margin)
    a_max:    np.ndarray  # shape (6,)  per-joint action range max (with margin)

    def save(self, path: Path) -> None: ...        # JSON with arrays as lists
    @classmethod
    def load(cls, path: Path) -> "Thresholds": ...
```

## Metric module contract

Every `stability_monitor/metrics/<name>.py` exposes:

```python
def batch(action: np.ndarray, state: np.ndarray, cfg: Config) -> np.ndarray: ...
class Streaming<Name>:
    def __init__(self, cfg: Config) -> None: ...
    def update(self, a_k: np.ndarray, q_k: np.ndarray) -> float | np.ndarray: ...
    def reset(self) -> None: ...
```

Conventions:

- Warmup frames (k < N-1) → `np.nan` in batch; `np.nan` from `update()` until
  enough samples have accumulated.
- Tracking error first frame undefined (offset alignment) → `np.nan`.
- `state` is unsmoothed at the boundary; SG smoothing happens inside metrics
  that need derivatives (jerk, SPARC).

Per-step streaming cost:

- tracking — O(1) running sum of squares.
- jerk — O(N) recompute over window (small N).
- SPARC — O(N log N) FFT over window.
- HF power — O(N) Welch over window.
- stall — O(1) running count (window-mean of an instant flag).

## Top-level monitor (`stability_monitor/monitor.py`)

```python
class StabilityMonitor:
    def __init__(self, cfg: Config, thresholds: Thresholds): ...
    def run_batch(self, action: np.ndarray, state: np.ndarray) -> pd.DataFrame: ...
    def step(self, a_k: np.ndarray, q_k: np.ndarray) -> MonitorOutput: ...
    def reset(self) -> None: ...
```

`MonitorOutput` (dataclass): the five m_ell scalars + per-joint diagnostics
(E_i^RMS, J_i^RMS, rho_i^HF, sigma_i) + raw flag + persisted flag.

## Intervention rule (`stability_monitor/intervention.py`)

```python
def any_of(m: np.ndarray, theta: np.ndarray) -> bool: ...
def weighted_sum(m: np.ndarray, theta: np.ndarray,
                 w: np.ndarray, theta_sigma: float) -> bool: ...

class PersistenceFilter:
    """Require P consecutive True values before declaring intervention."""
    def __init__(self, P: int) -> None: ...
    def update(self, raw: bool) -> bool: ...
    def reset(self) -> None: ...
```

## Loader (`stability_monitor/io/lerobot_loader.py`)

```python
@dataclass
class EpisodeArrays:
    dataset_id:    str
    episode_index: int
    action:        np.ndarray  # (T, 6) float32
    state:         np.ndarray  # (T, 6) float32
    timestamp:     np.ndarray  # (T,)   float64

def iter_episodes(
    dataset_id: str,
    only_successful: bool = False,
    labels_csv: Path | None = None,
) -> Iterator[EpisodeArrays]: ...
```

Pulls via `snapshot_download`, reads
`data/chunk-000/file-000.parquet`, groups by `episode_index`. If
`only_successful=True`, filters to episodes whose label is `success` in the
companion CSV.

## CLI defaults

- `scripts/calibrate.py --datasets nc8304/eval_smolvla-aug
  nc8304/eval_smolvla-phase-split-new-prompts
  nc8304/eval_smolvla-phase-split_combined --successful-only --out thresholds.json`
- `scripts/evaluate.py --dataset <id> --episode N --thresholds thresholds.json
  --out results.parquet`
- `scripts/visualize.py --results results.parquet --out results.png`

## Implementation order

Per spec §10, one metric at a time, end-to-end (batch → streaming → tests),
in this order so each new metric exercises the previous infrastructure:

1. tracking — establishes Config, EpisodeArrays, streaming pattern.
2. jerk — adds SG smoother + 5-point stencil; tests the causality lag.
3. sparc — first FFT-based metric.
4. hf_power — Welch on per-joint error.
5. stall — uses calibrated thresholds; first metric tied to Thresholds.

Then: calibration script → monitor composition → intervention rule →
evaluate/visualize CLIs → DESIGN.md cleanup → mypy/ruff/pyproject pinning.
