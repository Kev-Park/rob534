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

    for i in range(n_frames):
        frame = {}
        for k in data_keys:
            val = buf[k]
            if hasattr(val, "__len__") and i < len(val):
                frame[k] = val[i]
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
        single_task:    Task string written into both datasets.
        num_episodes:   Total episode cap across both policies combined.
        episode_time_s: Max seconds per episode before it auto-ends.
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

    counts = {"A": 0, "B": 0}  # episodes completed per policy
    current_label = "A"         # start on A; 'q' flips this each time

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
                        else:
                            # Normal end — save only to the active policy's dataset.
                            dataset.save_episode()
                    else:
                        print(f"  WARNING: Episode {i + 1} (Policy {label}) collected no frames — skipping save.")
                        if dataset.episode_buffer is not None:
                            dataset.clear_episode_buffer()

                    # ── FLIP POLICY IF q WAS PRESSED ──────────────────────────
                    if events["switch_policy"]:
                        current_label = "B" if current_label == "A" else "A"
                        print(f"  Now on Policy {current_label}.")
                    # Pass the held-position flag forward to next iteration
                    events["_held_position"] = events["switch_policy"]

                    if events["stop_recording"]:
                        break

    finally:
        # Park arm at home and clean up regardless of how the run ended.
        _go_home_with_robot(robot)
        robot.disconnect()
        listener.stop()
        switch_listener.stop()
        # finalize() writes index files and closes video writers.
        # Must be called on both so neither dataset is left incomplete.
        dataset_a.finalize()
        dataset_b.finalize()
