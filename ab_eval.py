"""
================================================================================
HOW TO USE ab_eval.py
================================================================================

WHAT IT DOES
------------
Runs two policies back to back on the same robot. Starts on policy A and stays
there until you manually press 'q' to switch. You control when the switch
happens — there is no automatic alternation. Data is saved to two separate
HuggingFace datasets so you can compare results later.

SETUP
-----
1. Plug the arm into COM5
2. Plug the camera in (USB, camera index 1)
3. Make sure you are logged into HuggingFace:
       huggingface-cli login
4. Edit the call at the bottom of better_code.py (or your own script):

       from ab_eval import do_ab_eval
       from better_code import resolve_policy_path

       do_ab_eval(
           policy_path_a=resolve_policy_path("SkywalkerLi/smolvla-phase-split"),
           policy_path_b=resolve_policy_path("SkywalkerLi/smolvla-aug"),
           repo_id_a="SkywalkerLi/eval_smolvla-phase-split",
           repo_id_b="SkywalkerLi/eval_smolvla-aug",
           num_episodes=20,       # total episodes across both policies
           episode_time_s=45,     # max seconds per episode
       )

   Change the model names and repo IDs to whatever you are testing.
   resolve_policy_path() checks if the model is already downloaded locally
   and skips the download if so.

STARTUP
-------
When you run it, both policies load into GPU memory before anything moves.
This takes about 20-30 seconds each. It only happens once — no reloading
between episodes.

KEYBOARD CONTROLS (while an episode is running)
-----------------------------------------------
    q        Switch to the other policy.
             - Ends the current episode immediately
             - Motors FREEZE at their exact position (no go_home)
             - The episode data is saved to BOTH datasets
             - Next episode starts with the other policy from the same position
             Use this when you want a direct comparison with the scene unchanged.

    d / →    End the episode early the normal way.
             - Arm goes back to home position before the next episode
             - Data saved only to the current policy's dataset
             - Same policy continues next episode

    Escape   Stop the whole run cleanly.
             - Arm goes home, datasets are finalized, robot disconnects

WHAT HAPPENS WHEN YOU PRESS q
------------------------------
Say policy A is running and you press q at 12 seconds:
  - Episode ends, 360 frames are in the buffer
  - Those 360 frames get copied into policy B's dataset too
  - Both datasets save the episode
  - Motors stay frozen — cube is still in the same spot
  - Next episode: policy B takes over from that exact position
  - Press q again to switch back to A

NORMAL EPISODE (no q press)
----------------------------
  - Episode runs until time limit (episode_time_s) or you press d
  - Arm goes home between episodes
  - Data saved only to the active policy's dataset
  - Same policy runs again next episode

WHERE DATA IS SAVED
-------------------
Locally at:
    C:\\Users\\<you>\\.cache\\huggingface\\lerobot\\<repo_id>\\

If you stop and restart, the script picks up where it left off automatically
(it checks for meta/tasks.parquet to detect a valid existing dataset).

TO UPLOAD AFTER YOU ARE DONE
-----------------------------
    python push_eval_to_hub.py <local_path> <hub_repo_id>

    Example:
    python push_eval_to_hub.py C:/Users/calle/.cache/huggingface/lerobot/SkywalkerLi/eval_smolvla-phase-split SkywalkerLi/eval_smolvla-phase-split

================================================================================
"""

import shutil
import subprocess
import threading
from pathlib import Path

# Helpers imported from better_code.py (must be in the same directory):
#   _check_starvation     — warns if CPU/RAM is under pressure before opening camera
#   _go_home_with_robot   — smoothly moves arm to the position saved in home_pos.json
#   _rerun_is_running     — checks port 9090 so we don't restart rerun if already open
#   _wait_and_open_viewer — waits for rerun server to start then opens the browser tab
#   repo_id_from_policy   — derives "SkywalkerLi/eval_<model>" from a policy path
from better_code import (
    _check_starvation,
    _go_home_with_robot,
    _rerun_is_running,
    _wait_and_open_viewer,
    repo_id_from_policy,
)


def _append_episode_stats_ab(csv_path: str, ep_idx: int, policy_label: str, ep_state) -> None:
    """Append one episode's stats row to a CSV (creates file + header on first write).

    Columns: episode_index, policy, then all EpisodeState dataclass fields.
    """
    import csv
    from dataclasses import asdict
    path = Path(csv_path)
    row = {"episode_index": ep_idx, "policy": policy_label, **asdict(ep_state)}
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  [stats] ep {ep_idx} (Policy {policy_label}) -> {csv_path}")


def _duplicate_buffer(source_dataset, target_dataset) -> int:
    """
    Copy every buffered frame from source_dataset into target_dataset.

    Called when 'q' is pressed so both datasets receive the episode that was
    mid-flight when the switch happened.

    IMPORTANT: must be called BEFORE source_dataset.save_episode() because
    save_episode() clears the buffer — calling this after would copy nothing.

    How it works:
      episode_buffer is a dict where each feature key (e.g. "action",
      "observation.state", "observation.images.camera1") maps to a list of
      per-frame values. We iterate frame-by-frame and call add_frame() on
      the target, which lets that dataset manage its own index counters.
      Metadata keys (frame_index, episode_index, timestamp, etc.) are skipped
      because add_frame() recomputes them — passing the source values in would
      cause index collisions between the two datasets.

    Returns number of frames copied (0 if the buffer was empty).
    """
    buf = source_dataset.episode_buffer
    if buf is None or buf.get("size", 0) == 0:
        return 0

    n_frames = buf["size"]

    # Skip keys that the dataset manages internally
    skip_keys = {"size", "frame_index", "episode_index", "index", "task_index", "timestamp"}
    data_keys = [k for k in buf if k not in skip_keys]

    import numpy as np
    from PIL import Image as PILImage

    for i in range(n_frames):
        frame = {}
        for k in data_keys:
            val = buf[k]
            if hasattr(val, "__len__") and i < len(val):
                v = val[i]
                # VideoEncodingManager stores image frames as file paths (strings).
                # Reload the actual pixel data before passing to the target dataset.
                # Only do this for image keys — other string values (e.g. "task") pass through as-is.
                if isinstance(v, str) and "image" in k:
                    v = np.array(PILImage.open(v))
                frame[k] = v
        if frame:
            target_dataset.add_frame(frame)

    return n_frames


def _start_switch_key_listener(events):
    """
    Starts a second background keyboard listener that watches only for 'q'.

    lerobot's init_keyboard_listener() already handles d/→ (exit early),
    a/← (rerecord), and Escape (stop recording). We add 'q' as a separate
    pynput listener on top — pynput supports multiple concurrent listeners
    on the same keyboard without conflict.

    When 'q' is pressed:
      - events["exit_early"] = True   → record_loop exits on its next iteration
      - events["switch_policy"] = True → do_ab_eval sees this after the episode
                                         and skips go_home (motors hold position)
                                         then flips current_label to the other policy

    Returns the started listener so the caller can stop it in a finally block.
    """
    from pynput import keyboard

    def on_press(key):
        try:
            if key.char == "q":
                events["exit_early"] = True
                events["switch_policy"] = True
                print("\n  [q] Policy switch — holding position, ending episode early.")
        except AttributeError:
            pass  # special keys (shift, ctrl, arrows) don't have .char

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def do_ab_eval(
    policy_path_a,
    policy_path_b,
    repo_id_a=None,
    repo_id_b=None,
    single_task="Grab the cube and drop it",
    num_episodes=20,
    episode_time_s=60,
    use_struggle_monitor=False,
    struggle_key_file=r"C:\Users\calle\Desktop\gem.txt",
    struggle_check_interval=2.0,
    struggle_threshold=0.6,
    auto_switch=False,
    stats_csv=None,
):
    """
    A/B policy eval driven by manual q-key switches.

    Starts on policy A and stays on it until you press 'q', which switches
    to policy B. Press 'q' again to switch back to A. You control when
    switches happen — the script never alternates automatically.

    Both policies are loaded once at startup and stay in GPU memory for the
    whole run. The robot and camera also stay connected throughout, so there
    is no warmup delay between episodes.

    Args:
        policy_path_a:  HF model ID or local snapshot path for policy A.
                        Use resolve_policy_path() to get the local path if cached.
        policy_path_b:  Same for policy B.
        repo_id_a:      Dataset repo_id for policy A results. Auto-derived from
                        policy_path_a if not given.
        repo_id_b:      Same for policy B.
        single_task:             Task string written into both datasets.
        num_episodes:            Total episode cap across both policies combined.
        episode_time_s:          Max seconds per episode before it auto-ends.
        use_struggle_monitor:    If True, runs a LiveStruggleMonitor in the background.
                                 The monitor polls Gemini every struggle_check_interval
                                 seconds. When the EMA score exceeds struggle_threshold,
                                 the episode ends early. Episode stats go to stats_csv.
        struggle_key_file:       Path to a file containing the Gemini API key.
        struggle_check_interval: Seconds between Gemini assessments.
        struggle_threshold:      EMA score above which is_struggling() triggers.
        auto_switch:             If True (requires use_struggle_monitor=True), the policy
                                 flips automatically when the monitor fires — no 'q' press
                                 needed. The arm holds position (never goes home) so the
                                 scene is identical for the other policy. Unlike 'q', only
                                 the active policy's dataset receives the episode (no
                                 buffer duplication to the other dataset).
        stats_csv:               Optional path to a CSV for per-episode stats.
                                 Appended to (not overwritten) so partial runs survive.
    """
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
    from lerobot.datasets.utils import combine_feature_dicts
    from lerobot.datasets.video_utils import VideoEncodingManager
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.processor import make_default_processors
    from lerobot.processor.rename_processor import rename_stats
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.scripts.lerobot_record import record_loop
    from lerobot.utils.control_utils import (init_keyboard_listener, sanity_check_dataset_name,
                                              sanity_check_dataset_robot_compatibility)
    from lerobot.utils.utils import init_logging
    from lerobot.utils.visualization_utils import init_rerun

    # Derive dataset repo_ids from policy paths if not given explicitly.
    # e.g. "SkywalkerLi/smolvla-aug" -> "SkywalkerLi/eval_smolvla-aug"
    if repo_id_a is None:
        repo_id_a = repo_id_from_policy(policy_path_a)
    if repo_id_b is None:
        repo_id_b = repo_id_from_policy(policy_path_b)

    # Warn if CPU/RAM looks stressed before we open the camera
    _check_starvation()

    # Start rerun visualization on port 9090. Skip if already running so we
    # don't clear the view from a previous session.
    if _rerun_is_running():
        print("  Rerun viewer already running on :9090, skipping restart.")
    else:
        subprocess.run(["taskkill", "/f", "/im", "rerun.exe"], capture_output=True)
        subprocess.Popen(["rerun", "--serve-web"])
    threading.Thread(target=_wait_and_open_viewer, daemon=True).start()

    init_logging()
    init_rerun(session_name="recording")

    # ── ONE-TIME HARDWARE + PIPELINE SETUP ───────────────────────────────────
    # Everything here is created once and reused across all episodes for both
    # policies. Previously we called lerobot as a subprocess per episode which
    # reloaded weights (~20-30s) and re-warmed the camera (~8s) every time.

    # SO-101 arm on COM5 + OpenCV camera at index 1.
    # camera name "camera1" must match observation.images.camera1 in the policy.
    robot_cfg = SOFollowerRobotConfig(
        port="COM5",
        id="student_arm",
        use_degrees=True,
        cameras={
            "camera1": OpenCVCameraConfig(
                index_or_path=1, fps=30, width=640, height=480, warmup_s=2,
            )
        },
    )
    robot = SOFollower(robot_cfg)

    # Stateless pipelines shared by both policies
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Dataset feature schema derived from the robot hardware (joint names, image shape, etc.)
    # Both datasets use the same schema since they record from the same robot.
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    def _load_cfg(policy_path):
        """Load policy config and fix any cluster-specific paths baked into it."""
        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.pretrained_path = policy_path
        cfg.device = "cuda"
        # Policies trained on a compute cluster may have an absolute path like
        # /scratch/gpfs/... baked into config.json for the VLM backbone.
        # Override it to the public HF model ID so it resolves locally.
        if hasattr(cfg, "vlm_model_name") and cfg.vlm_model_name.startswith("/"):
            cfg.vlm_model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        return cfg

    def _open_dataset(repo_id, policy_cfg):
        """
        Open or create a LeRobotDataset.
          - tasks.parquet exists       → valid dataset, resume (append episodes)
          - folder exists, no parquet  → incomplete/crashed run, delete and recreate
          - no folder                  → create fresh
        """
        dataset_path = Path.home() / ".cache/huggingface/lerobot" / repo_id
        resuming = (dataset_path / "meta" / "tasks.parquet").exists()
        if resuming:
            print(f"  [{repo_id}] Resuming existing dataset.")
            ds = LeRobotDataset(repo_id, root=dataset_path)
            ds.start_image_writer(num_processes=0, num_threads=4)
            sanity_check_dataset_robot_compatibility(ds, robot, 30, dataset_features)
        else:
            if dataset_path.exists():
                print(f"  [{repo_id}] Incomplete folder found, starting fresh.")
                shutil.rmtree(dataset_path)
            sanity_check_dataset_name(repo_id, policy_cfg)
            ds = LeRobotDataset.create(
                repo_id,
                fps=30,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=4,  # background threads write video without blocking control loop
            )
        return ds

    # ── LOAD BOTH POLICIES UPFRONT ────────────────────────────────────────────
    # SmolVLA-500M takes ~20-30s to load. Doing it here means zero switching
    # overhead between episodes — both models sit in VRAM ready to go.

    print("\n--- Policy A ---")
    cfg_a = _load_cfg(policy_path_a)
    dataset_a = _open_dataset(repo_id_a, cfg_a)
    print("  Loading policy A weights...")
    policy_a = make_policy(cfg_a, ds_meta=dataset_a.meta)
    pre_a, post_a = make_pre_post_processors(
        policy_cfg=cfg_a,
        pretrained_path=policy_path_a,
        dataset_stats=rename_stats(dataset_a.meta.stats, {}),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    print("\n--- Policy B ---")
    cfg_b = _load_cfg(policy_path_b)
    dataset_b = _open_dataset(repo_id_b, cfg_b)
    print("  Loading policy B weights...")
    policy_b = make_policy(cfg_b, ds_meta=dataset_b.meta)
    pre_b, post_b = make_pre_post_processors(
        policy_cfg=cfg_b,
        pretrained_path=policy_path_b,
        dataset_stats=rename_stats(dataset_b.meta.stats, {}),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    # Connect robot + camera once. Keyboard listener starts here too.
    robot.connect()
    listener, events = init_keyboard_listener()       # lerobot's default keys (d, Escape, etc.)
    switch_listener = _start_switch_key_listener(events)  # our 'q' key on top
    events["switch_policy"] = False
    events["auto_switched"] = False

    counts = {"A": 0, "B": 0}  # episodes completed per policy
    current_label = "A"         # start on A; 'q' flips this each time

    # ── Struggle monitor (optional) ───────────────────────────────────────────
    monitor = None
    if use_struggle_monitor:
        from struggle_monitor import LiveStruggleMonitor
        monitor = LiveStruggleMonitor(
            key_file=struggle_key_file,
            model="gemini-2.5-flash",
            check_interval=struggle_check_interval,
            interrupt_threshold=struggle_threshold,
        )
        monitor.start()
        monitor.start_capture(camera_index=1)
        print("  [StruggleMonitor] watching camera — will flag struggling episodes")

    def _struggle_watcher(stop_evt):
        """Background thread: sets exit_early when monitor signals struggling.

        With auto_switch=True, also sets auto_switched so the policy flips after
        the episode — the arm goes home rather than holding position (unlike 'q').
        """
        while not stop_evt.is_set():
            if monitor and monitor.is_struggling():
                if auto_switch:
                    print(f"\n  [StruggleMonitor] STRUGGLING — auto-switching policy after episode")
                    events["auto_switched"] = True
                else:
                    print(f"\n  [StruggleMonitor] STRUGGLING — exiting episode early")
                events["exit_early"] = True
                break
            stop_evt.wait(timeout=0.25)

    try:
        # Both VideoEncodingManagers stay open for the full run so their
        # background video-writing threads are always ready, regardless of
        # which dataset is currently active.
        with VideoEncodingManager(dataset_a):
            with VideoEncodingManager(dataset_b):
                for i in range(num_episodes):

                    # ── PICK ACTIVE POLICY ────────────────────────────────────
                    # current_label only changes when 'q' is pressed — it is NOT
                    # automatically alternated. Same policy keeps running until
                    # you manually switch.
                    label   = current_label
                    dataset = dataset_a if label == "A" else dataset_b
                    policy  = policy_a  if label == "A" else policy_b
                    pre     = pre_a     if label == "A" else pre_b
                    post    = post_a    if label == "A" else post_b
                    counts[label] += 1

                    # Reset events at the top of every episode so stray keypresses
                    # during go_home don't immediately exit the record_loop.
                    events["exit_early"] = False
                    events["rerecord_episode"] = False
                    events["switch_policy"] = False
                    events["auto_switched"] = False

                    if monitor:
                        monitor.reset_signal()

                    # Per-episode watcher thread: ends episode early if struggling.
                    watcher_stop = threading.Event()
                    if monitor:
                        watcher = threading.Thread(
                            target=_struggle_watcher, args=(watcher_stop,), daemon=True
                        )
                        watcher.start()

                    # ── GO HOME (skipped after q) ──────────────────────────────
                    # Normally the arm returns home so every episode starts from
                    # the same known pose. If the previous episode ended with 'q',
                    # _held_position is True and we skip go_home — the arm and
                    # the object stay exactly where they are so the other policy
                    # gets a fair attempt from the identical starting state.
                    held = events.get("_held_position", False)
                    if i == 0 or not held:
                        _go_home_with_robot(robot)

                    print(f"\n  Episode {i + 1}/{num_episodes} — "
                          f"Policy {label} (ep {counts[label]} for {label})")

                    # ── RUN EPISODE ───────────────────────────────────────────
                    # record_loop runs at 30 Hz: read obs -> policy forward pass
                    # -> send action -> store frame. Exits when control_time_s
                    # is reached or events["exit_early"] is set (d or q key).
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=30,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        policy=policy,
                        preprocessor=pre,
                        postprocessor=post,
                        dataset=dataset,
                        control_time_s=episode_time_s,
                        single_task=single_task,
                        display_data=True,
                    )

                    # Stop watcher thread and collect episode stats.
                    if monitor:
                        watcher_stop.set()
                        ep_state = monitor.get_episode_state()
                        print(f"  [EpisodeState] {ep_state}")

                    # ── SAVE EPISODE ──────────────────────────────────────────
                    # Guard against empty buffer — can happen if 'q' was pressed
                    # during go_home before the record_loop captured any frames.
                    frames_collected = (
                        dataset.episode_buffer is not None
                        and dataset.episode_buffer.get("size", 0) > 0
                    )
                    if frames_collected:
                        if events["switch_policy"]:
                            # 'q' pressed mid-episode: copy frames to the OTHER
                            # dataset before saving so both get the full trajectory.
                            # Must happen before save_episode() clears the buffer.
                            other_label   = "B" if label == "A" else "A"
                            other_dataset = dataset_b if label == "A" else dataset_a
                            n_copied = _duplicate_buffer(dataset, other_dataset)
                            print(f"  [switch] Saving {n_copied} frames to both datasets.")
                            dataset.save_episode()        # clears source buffer
                            other_dataset.save_episode()  # saves the copied frames
                            # Write stats for both datasets (same episode, both policies)
                            if stats_csv and monitor:
                                _append_episode_stats_ab(
                                    stats_csv, dataset.num_episodes - 1, label, ep_state)
                                _append_episode_stats_ab(
                                    stats_csv, other_dataset.num_episodes - 1, other_label, ep_state)
                        else:
                            # Normal end — save only to the active policy's dataset.
                            dataset.save_episode()
                            if stats_csv and monitor:
                                _append_episode_stats_ab(
                                    stats_csv, dataset.num_episodes - 1, label, ep_state)
                    else:
                        print(f"  WARNING: Episode {i + 1} (Policy {label}) collected no frames — skipping save.")
                        if dataset.episode_buffer is not None:
                            dataset.clear_episode_buffer()

                    # ── FLIP POLICY IF q WAS PRESSED OR MONITOR TRIGGERED ─────
                    if events["switch_policy"]:
                        current_label = "B" if current_label == "A" else "A"
                        print(f"  Now on Policy {current_label}.")
                    elif events.get("auto_switched"):
                        current_label = "B" if current_label == "A" else "A"
                        print(f"  [AutoSwitch] Monitor triggered — now on Policy {current_label}.")
                    # Hold position on any policy switch (q or auto) — arm never goes
                    # home between policies so the scene stays identical for fair comparison.
                    events["_held_position"] = events["switch_policy"] or bool(events.get("auto_switched"))

                    if events["stop_recording"]:
                        break

    finally:
        if monitor:
            monitor.stop()
        # Park arm at home and clean up regardless of how the run ended.
        _go_home_with_robot(robot)
        robot.disconnect()
        listener.stop()
        switch_listener.stop()
        # finalize() writes index files and closes video writers.
        # Must be called on both so neither dataset is left incomplete.
        dataset_a.finalize()
        dataset_b.finalize()


def simulate_ab_on_dataset(
    dataset_id: str | None = None,
    existing_stats_csv: str | None = None,
    key_file: str = r"C:\Users\calle\Desktop\gem.txt",
    model: str = "gemini-2.5-flash",
    check_interval_s: float = 2.0,
    window_seconds: float = 20.0,
    interrupt_threshold: float = 0.5,
    ema_alpha: float = 0.25,
    results_csv: str | None = None,
    out_csv: str = "ab_simulated_stats.csv",
    video_dir: str | None = None,
    max_episodes: int | None = None,
):
    """Simulate automatic A/B policy switching on a LeRobot dataset.

    Two modes:
      1. existing_stats_csv — load a CSV already produced by test_on_dataset
         (e.g. train_stats.csv). No Gemini calls; just replays the interrupt
         decisions and adds a policy label. Use this for quick offline testing.
      2. dataset_id — runs test_on_dataset from scratch (makes Gemini API calls),
         then simulates the switching on top.

    Switch rule: starts on policy "A". Each time final_interrupt is True (monitor
    would have triggered), the NEXT episode flips to the other policy. This mirrors
    auto_switch=True in do_ab_eval on the live robot.

    Writes out_csv with all original columns plus "policy" and "auto_switched_after".
    Prints per-policy stats (success rate, clean pickup/drop, mean interrupt prob).

    Args:
        dataset_id:          HuggingFace dataset ID. Used when no existing_stats_csv.
        existing_stats_csv:  Path to a prior test_on_dataset CSV. Skips Gemini calls.
        key_file:            Gemini API key file (only used when running fresh).
        model:               Gemini model (only used when running fresh).
        check_interval_s:    Seconds between checks (only used when running fresh).
        window_seconds:      Rolling buffer length in seconds (fresh runs only).
        interrupt_threshold: EMA threshold for final_interrupt decision.
        ema_alpha:           EMA smoothing factor (fresh runs only).
        results_csv:         Ground-truth labels CSV (fresh runs only).
        out_csv:             Where to write the enriched A/B CSV.
        video_dir:           If set, render 3-panel videos per episode (fresh only).
        max_episodes:        Cap on episodes to process (fresh runs only).
    """
    import pandas as pd

    if existing_stats_csv is not None:
        print(f"Loading existing stats from {existing_stats_csv} ...")
        df = pd.read_csv(existing_stats_csv)
    elif dataset_id is not None:
        from struggle_monitor import test_on_dataset
        raw_csv = out_csv + ".raw.csv"
        df = test_on_dataset(
            dataset_id=dataset_id,
            key_file=key_file,
            model=model,
            check_interval_s=check_interval_s,
            window_seconds=window_seconds,
            interrupt_threshold=interrupt_threshold,
            ema_alpha=ema_alpha,
            results_csv=results_csv,
            out_csv=raw_csv,
            video_dir=video_dir,
            max_episodes=max_episodes,
        )
    else:
        raise ValueError("Provide either dataset_id or existing_stats_csv.")

    if df.empty:
        print("No episodes to process.")
        return df

    # ── Simulate A/B switching ────────────────────────────────────────────────
    # Replay per-episode interrupt decisions and assign a policy label.
    # When an interrupt would have fired, the NEXT episode flips to the other policy.
    #
    # Interrupt decision: peak_ema_score >= interrupt_threshold (if the column exists)
    # or final_interrupt from the CSV (pre-computed with whatever threshold was used
    # during the original run). Using peak_ema_score lets you experiment with
    # different thresholds without re-running Gemini.
    use_peak = "peak_ema_score" in df.columns
    policy_labels: list[str] = []
    switched_after: list[bool] = []
    cur = "A"
    for _, row in df.iterrows():
        policy_labels.append(cur)
        if use_peak:
            triggered = float(row.get("peak_ema_score", 0.0)) >= interrupt_threshold
        else:
            triggered = bool(row.get("final_interrupt", False))
        switched_after.append(triggered)
        if triggered:
            cur = "B" if cur == "A" else "A"

    df = df.copy()
    df.insert(1, "policy", policy_labels)
    df.insert(2, "auto_switched_after", switched_after)

    df.to_csv(out_csv, index=False)
    print(f"\nSaved A/B simulated results to {out_csv}  ({len(df)} episodes)")

    # ── Per-policy summary ────────────────────────────────────────────────────
    total_switches = sum(switched_after)
    print(f"\n=== Simulated A/B Summary (threshold={interrupt_threshold}) ===")
    print(f"  Total auto-switches : {total_switches}")
    for pol in ["A", "B"]:
        sub = df[df["policy"] == pol]
        n = len(sub)
        if n == 0:
            continue
        n_success    = int(sub["target_reached"].sum())      if "target_reached"      in sub else "?"
        n_clean_pick = int((sub["clean_pickup"] == True).sum()) if "clean_pickup"     in sub else "?"
        n_clean_drop = int((sub["clean_drop"]   == True).sum()) if "clean_drop"       in sub else "?"
        n_switched   = int(sub["auto_switched_after"].sum())
        mean_p       = sub["mean_interrupt_prob"].mean()     if "mean_interrupt_prob" in sub else float("nan")
        print(
            f"  Policy {pol} : {n:2d} episodes"
            f"  success={n_success}/{n}"
            f"  clean_pick={n_clean_pick}/{n}"
            f"  clean_drop={n_clean_drop}/{n}"
            f"  switched_away={n_switched}"
            f"  mean_p={mean_p:.3f}"
        )

    return df


if __name__ == "__main__":
    import sys

    if "--simulate" in sys.argv:
        # Test the A/B switching simulation on the existing training stats.
        # No robot or Gemini calls needed — just replays interrupt decisions.
        simulate_ab_on_dataset(
            existing_stats_csv=r"C:\Users\calle\PycharmProjects\RobotArm\rob534\train_stats.csv",
            out_csv=r"C:\Users\calle\PycharmProjects\RobotArm\rob534\ab_simulated_stats.csv",
            interrupt_threshold=0.5,
        )
    else:
        # Live A/B eval on real robot.
        from better_code import resolve_policy_path

        do_ab_eval(
            policy_path_b=resolve_policy_path("SkywalkerLi/smolvla-phase-split-new-prompts"),
            policy_path_a=resolve_policy_path("SkywalkerLi/smolvla-aug"),
            repo_id_b="SkywalkerLi/eval_smolvla-phase-split-new-prompts_policyB",
            repo_id_a="SkywalkerLi/eval_smolvla-aug_policyA",
            num_episodes=10,
            episode_time_s=45,
            use_struggle_monitor=True,
            auto_switch=True,
            stats_csv=r"C:\Users\calle\PycharmProjects\RobotArm\rob534\ab_eval_stats.csv",
        )
