# `stability_monitor` — quick reference

Online stability monitor for SO-101 VLA rollouts. The math lives in
`../stability_monitoring_methods.tex`; this README is a navigation aid.

## The five metrics

All derive from `action` (commanded joint positions, `a[k] ∈ ℝ⁶`) and
`observation.state` (measured joint positions, `q[k] ∈ ℝ⁶`) at
`f_s = 30 Hz`. The combined "monitor vector" is

```
m[k] = [ E^RMS[k], J^RMS[k], -η_SPARC[k], ρ^HF[k], σ̄[k] ]
```

— sign-aligned so every channel is "larger means worse".

| # | code name | symbol | what it catches | needs `Thresholds`? |
|---|---|---|---|:---:|
| 1 | `E_RMS`     | `E^RMS`   | sustained command-vs-actual gap | no |
| 2 | `J_RMS`     | `J^RMS`   | jerky motion (earliest fire of the five) | no |
| 3 | `neg_SPARC` | `-η_SPARC`| spectrum roughness (amplitude-invariant) | no |
| 4 | `rho_HF`    | `ρ^HF`    | high-frequency chatter in tracking error | no |
| 5 | `sigma_bar` | `σ̄`       | per-joint stall and joint-limit clip events | **yes** |

Each metric is in `metrics/<name>.py` and exposes a `batch(action, state, cfg)`
function plus a `StreamingX` class. Metric 5 takes an extra `thresholds`
argument because the stall and clip predicates need calibrated parameters.

## Calibrated thresholds (`Thresholds`)

Five fields written by `scripts/calibrate.py`. See `calibration.py`.

| field | shape | role |
|---|---|---|
| `theta`   | `(5,)` | **intervention thresholds** — one per `m`-vector channel |
| `delta_a` | `(6,)` | per-joint command-magnitude scale (input to σ_i^stall) |
| `delta_q` | `(6,)` | per-joint state-motion scale (input to σ_i^stall) |
| `a_min`   | `(6,)` | per-joint commanded-position lower limit (input to σ_i^clip) |
| `a_max`   | `(6,)` | per-joint commanded-position upper limit (input to σ_i^clip) |

The `delta_*` and `a_*` fields are **inputs to the stall metric** — they
parameterise the predicates that produce `σ̄[k]`. The intervention rule
itself only ever compares `m[k]` against `theta`.

## Using thresholds at runtime

```python
from stability_monitor.calibration import Thresholds
from stability_monitor.config import Config
from stability_monitor.metrics import tracking, jerk, sparc, hf_power, stall
import numpy as np

cfg = Config()
thr = Thresholds.load("thresholds/eval_smolvla-phase-split_combined.json")

# Compute the five m-channels. The first four take only (action, state);
# stall additionally takes `thr` because it needs delta_*, a_min, a_max.
m_E    = tracking.batch(action, state, cfg).agg
m_J    = jerk.batch(action, state, cfg).agg
m_negS = -sparc.batch(action, state, cfg)
m_HF   = hf_power.batch(action, state, cfg).agg
m_sig  = stall.batch(action, state, cfg, thr).agg

m = np.stack([m_E, m_J, m_negS, m_HF, m_sig], axis=1)  # (T, 5)

# Any-of intervention rule from the methods doc.
fired_per_channel = m > thr.theta            # (T, 5) bool
intervene_any     = fired_per_channel.any(axis=1)
```

Streaming mode is the same shape, with each metric exposed as a
`StreamingX` class whose `update(a_k, q_k)` returns the latest finite value.

## Why `theta[4] = 0`

Calibration has a chicken-and-egg: to set `theta[4]` (the intervention
threshold for `σ̄`) the conventional way, you'd take the 95th percentile
of `σ̄` on the success corpus — but computing `σ̄` requires `delta_*` and
`a_min/a_max`, which don't exist yet. Spec §6 resolves this by running
the `theta`-calibration pass with **placeholder** stall thresholds (huge
`delta_a`, zero `delta_q`, ±huge clip range), so the stall and clip
predicates can never fire and `σ̄ ≡ 0` on the success corpus. The 95th
percentile of zeros is zero, hence `theta[4] = 0`.

Semantically this encodes a stricter rule for `σ̄` than for the other
four channels:

- `E^RMS`, `J^RMS`, `-η_SPARC`, `ρ^HF`: `theta_l = P95` ⇒ *"channel `l`
  naturally hits this magnitude on 5 % of success-corpus frames; flag
  only when higher."*
- `σ̄`: `theta[4] = 0` ⇒ *"stall/clip events shouldn't happen at all on
  a successful run; flag any event."*

That's consistent with the physical meaning of the stall/clip predicates
— they detect concrete failure modes (joint pinned at load limit, action
saturating). The persistence filter (`P = 3`, ≈100 ms) still applies, so
a single-frame `σ̄ > 0` doesn't trigger; you need three consecutive
frames. If you'd rather tolerate some `σ̄`, edit `theta[4]` after
loading.

## Output formats (`thresholds/<name>.{json,csv}`)

- **`<name>.json`** — canonical, machine-loadable via `Thresholds.load()`.
- **`<name>.csv`** — long format `section, parameter, channel, value`
  with three sections:
  - `meta`: corpus identification (dataset name, episode count).
  - `threshold`: the fitted `Thresholds` object — `channel` is an
    `m`-vector name (for `theta`) or a joint name (for `delta_*`,
    `a_min`, `a_max`).
  - `summary`: pooled distribution of each `m`-channel on the success
    corpus (`n_finite`, `median`, `p95`, `p99`, `max`). By construction
    `summary p95` for channel `c` equals `threshold theta c`.

## Configuration (`Config`)

Defaults in `config.py` match the SO-101 setup:

| field | default | meaning |
|---|---:|---|
| `fs`            | 30 Hz   | sampling rate |
| `N`             | 30      | sliding-window length (≈1 s) |
| `M`             | 5       | stall look-back (≈167 ms) |
| `P`             | 3       | persistence-filter length (≈100 ms) |
| `sg_window, sg_polyorder` | 7, 3 | Savitzky-Golay smoother |
| `omega_h`       | 5 Hz    | HF-power-ratio crossover |
| `omega_c_max`   | 12 Hz   | SPARC adaptive-cutoff cap |
| `V_bar`         | 0.05    | SPARC normalised-spectrum threshold |
| `eps_clip_frac` | 0.02    | clip tolerance as fraction of joint range |

Joint order: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper`.
