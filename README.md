# rob534

ROB534 final project @ Princeton.

## Dataset

Our data is hosted on Hugging Face:

https://huggingface.co/datasets/nc8304/so101

---

## Interrupt Monitor (`struggle_monitor.py`)

> **Calibration status — work in progress**
>
> The current interrupt threshold (EMA ≥ 0.5) and scaling parameters (time ramp,
> probability cap, median filter) were tuned on the **first 10 training episodes only**.
> Across those 10 episodes the peak EMA ranged from 0.03 to 0.44, giving
> 0/10 false positives on training data and 80% sensitivity / 100% specificity on
> 10 labeled eval episodes.
>
> **To improve calibration:** run the full 80-episode training set through the batch
> script and set the interrupt threshold at the 95th percentile of `peak_ema_score`
> across all training episodes. This gives a principled, data-driven threshold instead
> of the current hand-tuned value.
>
> **Cost:** ~70 additional Gemini API calls per episode × 80 episodes — feasible with
> `gemini-2.5-flash` (cheap), but a full `gemini-2.5-pro` calibration run will be
> noticeably more expensive. Use `--model gemini-2.5-flash` for the calibration sweep
> and reserve `gemini-2.5-pro` for final eval runs where accuracy matters most.

### Motivation

Imitation-learning policies for robot manipulation can fail silently. When a
pick-and-place policy starts looping, drops the block repeatedly, or gets stuck,
a human operator needs to notice and switch to a different policy. Detecting this
automatically with classical heuristics alone is hard: jerk and Lyapunov metrics
flag erratic motion but lack the semantic understanding to tell apart a hesitant
but ultimately successful grasp from a genuine failure.

We use Gemini as a vision-language supervisor. Every few seconds it receives a
short clip of the episode alongside supplementary sensor data, and returns an
interrupt probability that drives an automatic switch signal.

---

### How interrupt probability is calculated

Every 2 seconds, 12 frames evenly sampled from the last 20 seconds are sent to
Gemini alongside two supplementary text blocks derived from the robot's sensor
stream:

- **Stability block** — rolling jerk and a Lyapunov-proxy trend computed from
  joint actions and observation states.
- **Gripper block** — raw gripper-angle trace with numerically counted open/close
  events. This is supplementary: Gemini is asked to count pickup and drop attempts
  from the *video*; the sensor trace lets it cross-check its visual read.

Gemini watches the frames, counts pickup and drop attempts, and returns a raw
`interrupt_probability` (0–1) with a one-sentence reason.

### Post-processing pipeline

The raw score passes through four stages before driving the interrupt decision:

1. **Time ramp** — multiplied by `min(1, t / 30s)`. At episode start the score
   is zeroed out and reaches full weight only after 30 seconds. Prevents the
   initial open-gripper approach from triggering a false positive.
2. **Probability cap** — clipped to a ceiling that starts at 0.75 and rises
   linearly to 0.90 between 30 s and 60 s. High probabilities are only possible
   after the arm has had sustained time to fail.
3. **Median filter** — rolling median over the last 3 checks kills isolated spike
   assessments. One bad Gemini call cannot trigger an interrupt on its own.
4. **EMA** — exponential moving average (`α = 0.25` by default) smooths the
   trend: `score = α × p_filtered + (1−α) × score`. `is_struggling()` fires when
   this score exceeds the threshold (default 0.5).

### Why probabilities are lower on training data

Probabilities tend to be systematically lower on training data because training
sets consist of curated, successful demonstrations — the arm moves deliberately,
grasps cleanly, and completes the task in one or two attempts — exactly the
pattern Gemini is instructed to score near zero. Eval rollouts contain policy
failures and retries that are visually distinct (repeated gripper opens, erratic
corrections, diverging Lyapunov trend), so Gemini reliably assigns higher
probabilities to them, which is what gives the monitor its discriminative power.

---

### Annotated video output

Each episode can be rendered as a 3-panel video (`--video-dir`). The panels from
top to bottom are:

#### Top strip — action jerk

- Y-axis is the magnitude of the change in joint actions between consecutive
  steps (jerk). Higher = more erratic motion.
- Each segment is coloured by the Lyapunov trend at that moment: **green** =
  state moving *toward* the goal (converging), **red** = state moving *away*
  (diverging).
- Orange dashed verticals mark frames where jerk spiked more than 3 standard
  deviations above the episode mean — sudden lurches or slips.
- The blue vertical marked **"peak interrupt prob"** is the single 2-second check
  where the smoothed interrupt score was highest across the whole episode — the
  moment Gemini was most convinced the policy had failed.

#### Middle — camera feed

- Raw camera footage with a HUD overlay in the top-left corner showing:
  - `p` — current filtered interrupt probability (colour: green → red).
  - `ema` — current EMA struggle score (green below threshold, red above).
  - Time-ramp and cap indicators while active (`×factor ≤cap`).
  - Cumulative pickup and drop attempt counts.
  - Gemini's reason text, revealed progressively after each check.

#### Bottom strip — gripper angle

- Y-axis is the gripper opening angle in degrees. The orange dashed horizontal
  is the threshold (default 20°) above which the gripper is considered closed.
- The blue line is the raw signal; the **green shaded region** shows every frame
  where the gripper is holding (above threshold).
- **Lime vertical** = first detected pickup frame (gripper closes on the block).
- **Red vertical** = first detected drop frame (gripper opens to release).

#### Cyan playhead

The vertical cyan line in both strips tracks the current video frame, so you can
correlate what you see in the camera feed with where you are in the jerk and
gripper traces.

