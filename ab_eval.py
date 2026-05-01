"""
A/B policy evaluation — alternates between two policies every episode,
keeping data in separate datasets.

PURPOSE
-------
Compare two trained policies head-to-head on the same task. Each policy
runs on alternating episodes (A, B, A, B, ...) so conditions stay consistent.
Results are saved to two independent HuggingFace datasets that can be uploaded
and compared separately.

TYPICAL USE
-----------
    from ab_eval import do_ab_eval
    from better_code import resolve_policy_path

    do_ab_eval(
        policy_path_a=resolve_policy_path("SkywalkerLi/smolvla-phase-split"),
        policy_path_b=resolve_policy_path("SkywalkerLi/smolvla-aug"),
        repo_id_a="SkywalkerLi/eval_smolvla-phase-split",
        repo_id_b="SkywalkerLi/eval_smolvla-aug",
        num_episodes_each=10,   # 20 episodes total: A,B,A,B,...
        episode_time_s=45,
    )

KEYBOARD SHORTCUTS (during an episode)
---------------------------------------
    q        — freeze motors at current position and immediately switch to the
               other policy. Skips go_home so the physical scene (cube position,
               arm pose) stays unchanged for a fair A/B comparison.
    d / →    — end episode early the normal way (arm will go home before next ep)
    Escape   — stop the entire run cleanly

DEPENDENCIES
------------
Helper functions (_check_starvation, _go_home_with_robot, etc.) are imported
from better_code.py which must be in the same directory.
"""

import shutil
import subprocess
import threading
from pathlib import Path

# Helpers shared with better_code.py:
#   _check_starvation    — warns if CPU/RAM is under pressure before opening camera
#   _go_home_with_robot  — smoothly moves arm to saved home_pos.json position
#   _rerun_is_running    — checks :9090 so we don't restart rerun if already open
#   _wait_and_open_viewer — waits for rerun server then opens the browser tab
#   repo_id_from_policy  — derives "SkywalkerLi/eval_<model>" from a policy path
from better_code import (
    _check_starvation,
    _go_home_with_robot,
    _rerun_is_running,
    _wait_and_open_viewer,
    repo_id_from_policy,
)


def _duplicate_buffer(source_dataset, target_dataset) -> int:
    """
    Copy every buffered frame from source_dataset into target_dataset.

    WHY THIS EXISTS
    ---------------
    When the user presses 'q' mid-episode, the arm freezes and we switch
    policies — but the episode data collected so far only lives in the
    active dataset's episode_buffer. To ensure BOTH datasets get the full
    trajectory (as the user requested), we call this before save_episode()
    to replay the frames into the other dataset. Since VideoEncodingManager
    is open for both datasets, the video frames will be encoded correctly
    for each dataset independently.

    TIMING IS CRITICAL
    ------------------
    This MUST be called before save_episode() on the source. save_episode()
    clears the episode_buffer, so any call after that will copy nothing.

    HOW IT WORKS
    ------------
    episode_buffer is a dict where each feature key maps to a list of
    per-frame values (tensors, arrays, etc.), plus a "size" counter.
    We skip the dataset-managed metadata keys (frame_index, episode_index,
    index, task_index, timestamp) because add_frame() recomputes those
    internally — passing them in would cause index collisions between the
    two datasets.

    Args:
        source_dataset: The dataset that recorded the episode.
        target_dataset: The dataset to duplicate the frames into.

    Returns:
        Number of frames copied (0 if buffer was empty).
    """
    buf = source_dataset.episode_buffer
    if buf is None or buf.get("size", 0) == 0:
        return 0

    n_frames = buf["size"]

    # These keys are recomputed by add_frame() — don't pass them in or the
    # target dataset's internal counters will conflict with the source's values.
    skip_keys = {"size", "frame_index", "episode_index", "index", "task_index", "timestamp"}
    data_keys = [k for k in buf if k not in skip_keys]

    for i in range(n_frames):
        frame = {}
        for k in data_keys:
            val = buf[k]
            # Guard: make sure the list is long enough (should always be true,
            # but defensive in case of a partial/interrupted write).
            if hasattr(val, "__len__") and i < len(val):
                frame[k] = val[i]
        if frame:
            target_dataset.add_frame(frame)

    return n_frames


def _start_switch_key_listener(events):
    """
    Starts a second background keyboard listener that watches only for 'q'.

    Why a second listener?
    ----------------------
    lerobot's init_keyboard_listener() already listens for d/→ (exit early),
    a/← (rerecord), and Escape (stop). We can't easily inject 'q' into that
    listener, but pynput supports multiple concurrent listeners on the same
    keyboard without conflict — each receives all key events independently.

    What 'q' does:
    --------------
    1. Sets events["exit_early"] = True  → lerobot's record_loop exits on the
       next iteration (same effect as pressing 'd')
    2. Sets events["switch_policy"] = True  → our loop in do_ab_eval reads this
       after the episode ends and skips _go_home_with_robot, so the arm stays
       exactly where it is. The scene (cube, arm pose) remains unchanged for
       the next policy to attempt from the same starting state.

    Args:
        events: The shared events dict from lerobot's init_keyboard_listener().
                We write into it so both listeners share the same state.

    Returns:
        A started pynput.keyboard.Listener. Call .stop() in a finally block.
    """
    from pynput import keyboard

    def on_press(key):
        try:
            if key.char == "q":
                events["exit_early"] = True       # ends the current record_loop
                events["switch_policy"] = True    # tells our loop to skip go_home
                print("\n  [q] Policy switch requested — holding position, ending episode early.")
        except AttributeError:
            pass  # special keys (shift, ctrl, arrows) raise AttributeError on .char

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def do_ab_eval(
    policy_path_a,
    policy_path_b,
    repo_id_a=None,
    repo_id_b=None,
    single_task="Grab the cube and drop it",
    num_episodes_each=5,
    episode_time_s=60,
):
    """
    Alternating A/B policy eval. Runs policy A on even episodes and policy B
    on odd episodes, saving to two separate datasets.

    Both policies are loaded ONCE at startup and kept in GPU memory for the
    entire run — no reloading between episodes. The robot and camera are also
    connected once and stay open throughout.

    Episode order: A, B, A, B, ...  (total = num_episodes_each * 2)

    go_home behaviour:
    ------------------
    - Normally the arm returns to its saved home position (home_pos.json)
      between every episode so each attempt starts from the same pose.
    - If 'q' was pressed to end the previous episode, go_home is SKIPPED.
      This lets you compare both policies from identical arm/cube positions.

    Resume behaviour:
    -----------------
    Both datasets support resuming. If a dataset folder already contains
    meta/tasks.parquet it is considered complete and new episodes are appended.
    Incomplete folders (no tasks.parquet) are deleted and recreated fresh.

    Args:
        policy_path_a:     HF model ID ("SkywalkerLi/smolvla-phase-split") or
                           absolute local snapshot path. Use resolve_policy_path()
                           from better_code.py to get the local path if cached.
        policy_path_b:     Same as above for the second policy.
        repo_id_a:         HuggingFace dataset repo_id for policy A results.
                           Auto-derived from policy_path_a if None
                           (e.g. "SkywalkerLi/smolvla-phase-split" ->
                                 "SkywalkerLi/eval_smolvla-phase-split").
        repo_id_b:         Same for policy B.
        single_task:       Task description string embedded in both datasets
                           (shown in the lerobot viewer and stored in tasks.parquet).
        num_episodes_each: How many episodes to collect per policy.
                           Total episodes = num_episodes_each * 2.
        episode_time_s:    Maximum seconds per episode before auto-ending.
    """
    # Lazy lerobot imports — keeps startup fast and avoids importing GPU libs
    # until they are actually needed.
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

    # Derive dataset repo_ids from policy paths if not provided explicitly.
    # Example: "SkywalkerLi/smolvla-aug" -> "SkywalkerLi/eval_smolvla-aug"
    if repo_id_a is None:
        repo_id_a = repo_id_from_policy(policy_path_a)
    if repo_id_b is None:
        repo_id_b = repo_id_from_policy(policy_path_b)

    # Warn if CPU/RAM is under pressure — a busy system can cause the camera
    # capture loop to fall behind real-time and drop frames.
    _check_starvation()

    # Start the rerun visualization server (port 9876 gRPC / 9090 web).
    # We check first so we don't kill and restart it if it's already running
    # from a previous session — restarting clears the view history.
    if _rerun_is_running():
        print("  Rerun viewer already running on :9090, skipping restart.")
    else:
        subprocess.run(["taskkill", "/f", "/im", "rerun.exe"], capture_output=True)
        subprocess.Popen(["rerun", "--serve-web"])
        # Open the browser tab in the background while the rest of setup continues.
        threading.Thread(target=_wait_and_open_viewer, daemon=True).start()

    init_logging()
    init_rerun(session_name="recording")

    # ── ONE-TIME SETUP ────────────────────────────────────────────────────────
    # Robot, camera, and data processors are created once and shared across
    # all episodes for both policies. This avoids the ~8s camera warmup and
    # serial port reconnect that would happen if we restarted per episode.

    # SO-101 follower arm on COM5 with one OpenCV camera named "camera1".
    # warmup_s=2: camera stabilises for 2s after connect (was 10s, reduced).
    # The camera name "camera1" must match what the SmolVLA policy expects in
    # its observation keys (observation.images.camera1).
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

    # make_default_processors() returns three pipeline objects:
    #   teleop_action_processor    — normalises/formats actions from teleop
    #   robot_action_processor     — sends actions to the physical robot
    #   robot_observation_processor — reads observations (joint pos + images)
    # These are stateless and shared between both policies.
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Build the feature schema for the datasets (joint names, image shapes, etc.)
    # from the robot's actual hardware config. Both datasets use the same schema
    # since they're collecting from the same robot.
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
        """Load policy config from HF or local snapshot and apply runtime overrides."""
        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.pretrained_path = policy_path
        cfg.device = "cuda"
        # Policies trained on a compute cluster may have an absolute cluster path
        # baked into config.json for the VLM backbone (e.g. /scratch/gpfs/...).
        # Override it to the public HF model ID so transformers can find it locally.
        if hasattr(cfg, "vlm_model_name") and cfg.vlm_model_name.startswith("/"):
            cfg.vlm_model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        return cfg

    def _open_dataset(repo_id, policy_cfg):
        """
        Open (or create) a LeRobotDataset for the given repo_id.

        Resume logic:
          - tasks.parquet exists  → dataset is valid, append new episodes
          - folder exists but no tasks.parquet → partial/failed run, delete and recreate
          - folder does not exist → create fresh
        """
        dataset_path = Path.home() / ".cache/huggingface/lerobot" / repo_id
        resuming = (dataset_path / "meta" / "tasks.parquet").exists()
        if resuming:
            print(f"  [{repo_id}] Resuming existing dataset.")
            ds = LeRobotDataset(repo_id, root=dataset_path)
            ds.start_image_writer(num_processes=0, num_threads=4)
            # Verify the dataset's feature schema still matches the current robot config.
            sanity_check_dataset_robot_compatibility(ds, robot, 30, dataset_features)
        else:
            if dataset_path.exists():
                # Incomplete folder from a previous crashed run — start fresh.
                print(f"  [{repo_id}] Incomplete folder found, starting fresh.")
                shutil.rmtree(dataset_path)
            sanity_check_dataset_name(repo_id, policy_cfg)
            ds = LeRobotDataset.create(
                repo_id,
                fps=30,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=True,
                # image_writer_threads=4: background threads encode video frames
                # without blocking the main control loop.
                image_writer_processes=0,
                image_writer_threads=4,
            )
        return ds

    # ── LOAD BOTH POLICIES ────────────────────────────────────────────────────
    # SmolVLA-500M takes ~20-30s to load weights from disk into VRAM.
    # Loading both up front means zero switching overhead between episodes.

    print("\n--- Policy A ---")
    cfg_a = _load_cfg(policy_path_a)
    dataset_a = _open_dataset(repo_id_a, cfg_a)
    print("  Loading policy A weights (this takes ~20-30s)...")
    policy_a = make_policy(cfg_a, ds_meta=dataset_a.meta)
    # make_pre_post_processors builds the input normalisation and output
    # denormalisation pipelines specific to this policy's training stats.
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
    print("  Loading policy B weights (this takes ~20-30s)...")
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

    # Connect robot and camera once — stays open for all episodes.
    robot.connect()

    # init_keyboard_listener() sets up lerobot's default key bindings:
    #   d / →   sets events["exit_early"] = True
    #   a / ←   sets events["rerecord_episode"] = True
    #   Escape  sets events["stop_recording"] = True
    listener, events = init_keyboard_listener()

    # Our second listener adds 'q' on top of lerobot's defaults.
    # It writes into the same events dict so everything stays in sync.
    switch_listener = _start_switch_key_listener(events)
    events["switch_policy"] = False   # initialise our custom flag

    total_episodes = num_episodes_each * 2
    counts = {"A": 0, "B": 0}  # track how many episodes each policy has done

    try:
        # Open video encoding managers for BOTH datasets before starting.
        # VideoEncodingManager starts a background thread that writes MP4 frames
        # to disk. We need both open simultaneously because we don't know in
        # advance which dataset will receive the next episode.
        with VideoEncodingManager(dataset_a):
            with VideoEncodingManager(dataset_b):
                for i in range(total_episodes):

                    # ── SELECT POLICY FOR THIS EPISODE ────────────────────────
                    # Simple alternation: even index -> A, odd index -> B.
                    # This gives an interleaved sequence regardless of how many
                    # episodes are collected, which helps control for any drift
                    # in the physical setup over time.
                    label   = "A" if i % 2 == 0 else "B"
                    dataset = dataset_a if label == "A" else dataset_b
                    policy  = policy_a  if label == "A" else policy_b
                    pre     = pre_a     if label == "A" else pre_b
                    post    = post_a    if label == "A" else post_b
                    counts[label] += 1

                    # Clear all events at the start of each episode.
                    # This prevents a stray keypress during go_home (or any
                    # leftover True from the previous episode) from immediately
                    # exiting the very first frame of record_loop.
                    events["exit_early"] = False
                    events["rerecord_episode"] = False
                    events["switch_policy"] = False

                    # ── GO HOME (conditional) ──────────────────────────────────
                    # Normal path: move arm to saved home position so every
                    # episode starts from the same known configuration.
                    #
                    # Exception: if the previous episode ended with 'q', the
                    # _held_position flag will be True. In that case we skip
                    # go_home so the arm (and the object in the scene) remain
                    # exactly where they were — giving the next policy a fair
                    # chance to attempt from the identical starting state.
                    held = events.get("_held_position", False)
                    if i == 0 or not held:
                        _go_home_with_robot(robot)

                    print(f"\n  Episode {i + 1}/{total_episodes} — "
                          f"Policy {label} ({counts[label]}/{num_episodes_each})")

                    # ── RUN THE EPISODE ───────────────────────────────────────
                    # record_loop runs the closed-loop policy control at ~30 Hz:
                    #   1. Read observation (joint positions + camera frame)
                    #   2. Preprocess observation (normalise, move to GPU)
                    #   3. Run policy forward pass -> action chunk
                    #   4. Postprocess + send action to robot
                    #   5. Store frame in dataset buffer
                    #   6. Repeat until control_time_s elapsed or exit_early set
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

                    # ── SAVE EPISODE ──────────────────────────────────────────
                    # Guard against empty buffer. This can happen if exit_early
                    # was set before the first frame was captured (e.g. 'q' was
                    # pressed during the go_home movement right before the loop).
                    frames_collected = (
                        dataset.episode_buffer is not None
                        and dataset.episode_buffer.get("size", 0) > 0
                    )
                    if frames_collected:
                        if events["switch_policy"]:
                            # 'q' ended this episode mid-run. The user wants the
                            # full trajectory saved to BOTH datasets so that each
                            # policy's eval set contains the same episode for a
                            # fair comparison.
                            #
                            # We copy the buffer into the OTHER dataset FIRST
                            # (before save_episode clears it), then save both.
                            other_label   = "B" if label == "A" else "A"
                            other_dataset = dataset_b if label == "A" else dataset_a
                            n_copied = _duplicate_buffer(dataset, other_dataset)
                            print(f"  [switch] Duplicating {n_copied} frames -> dataset {other_label}.")
                            dataset.save_episode()       # clears buffer on source
                            other_dataset.save_episode() # saves the duplicated frames
                        else:
                            # Normal end (time limit or 'd'): save only to the
                            # active dataset as usual.
                            dataset.save_episode()
                    else:
                        print(f"  WARNING: Episode {i + 1} (Policy {label}) collected no frames — skipping save.")
                        if dataset.episode_buffer is not None:
                            dataset.clear_episode_buffer()

                    # ── CARRY SWITCH FLAG TO NEXT ITERATION ───────────────────
                    # Store whether 'q' ended this episode. The next iteration
                    # reads _held_position to decide whether to skip go_home.
                    events["_held_position"] = events["switch_policy"]

                    # Escape key: stop the whole run cleanly.
                    if events["stop_recording"]:
                        break

    finally:
        # Always park the arm at home before disconnecting, even if an exception
        # occurred. This avoids leaving motors in a random pose under torque.
        _go_home_with_robot(robot)
        robot.disconnect()
        listener.stop()        # stop lerobot's keyboard listener
        switch_listener.stop() # stop our 'q' listener
        # finalize() writes the dataset index files and closes the video writers.
        # Must be called on BOTH datasets so neither is left in a partial state.
        dataset_a.finalize()
        dataset_b.finalize()
