"""
================================================================================
HOW TO USE student_teacher_failure_analysis.py
================================================================================

WHAT IT DOES
------------
Runs N rollouts of a "student" policy. After each rollout, sends a contact
sheet of the episode to Gemini, which classifies the rollout as one of:

    success                                 → continue, no teacher needed
    failure_to_pick_up_block                → teacher runs with prompt
                                              "pick up the cube"
    failure_to_move_block_over_correct_region → teacher runs with prompt
                                                "bring cube over to the target"
    failure_to_drop_block                   → teacher runs with prompt
                                              "drop cube in the target region"

The teacher resumes from the student's final pose (motors hold position) so
it can attempt recovery from the actual failure state. Two HuggingFace
datasets are saved — one for student rollouts, one for teacher rollouts.

SETUP
-----
1. Plug the arm into the right port and the USB camera in (camera index 1).
2. `huggingface-cli login`
3. `export GEMINI_API_KEY=...`
4. Edit the call at the bottom of better_code.py:

       from student_teacher_failure_analysis import do_student_teacher_failure_analysis
       from better_code import resolve_policy_path

       do_student_teacher_failure_analysis(
           student_policy_path=resolve_policy_path("SkywalkerLi/smolvla-aug"),
           teacher_policy_path=resolve_policy_path("SkywalkerLi/smolvla-phase-split"),
           repo_id_student="SkywalkerLi/eval_st_student",
           repo_id_teacher="SkywalkerLi/eval_st_teacher",
           num_student_episodes=20,
           episode_time_s=45,
       )

KEYBOARD CONTROLS
-----------------
    d / →       End the current episode early (saves and moves on).
    Escape      Stop the whole run cleanly after the current episode.

GEMINI INPUT FORMAT
-------------------
By default a 4×4 grid of 16 evenly-spaced frames is sent as a single inline
JPEG. To fall back to sending a full MP4 instead, pass `use_video=True`.

================================================================================
"""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from better_code import (
    _check_starvation,
    _go_home_with_robot,
    _rerun_is_running,
    _wait_and_open_viewer,
    repo_id_from_policy,
)


OUTCOME_TO_TASK = {
    "failure_to_pick_up_block":                  "pick up the cube",
    "failure_to_move_block_over_correct_region": "bring cube over to the target",
    "failure_to_drop_block":                     "drop cube in the target region",
}


GRID_PROMPT_PREFIX = """You are looking at a single image that contains a grid of evenly-spaced
frames sampled from one robot rollout video. The frames are arranged in
row-major order: the TOP-LEFT tile is the earliest frame in time, moving
across each row left-to-right, then down to the next row, with the
BOTTOM-RIGHT tile being the latest frame. Treat the grid as a time
sequence, not as separate images.

"""


def _episode_has_frames(dataset) -> bool:
    return (
        dataset.episode_buffer is not None
        and dataset.episode_buffer.get("size", 0) > 0
    )


def _build_contact_sheet(image_entries, n: int = 16, cols: int = 4,
                         thumb_w: int = 320, thumb_h: int = 240,
                         jpeg_quality: int = 85) -> bytes:
    """Tile evenly-spaced frames into a grid JPEG. Returns the encoded bytes."""
    import numpy as np
    from PIL import Image

    if not image_entries:
        raise ValueError("Cannot build contact sheet: no frames in episode buffer.")

    n = min(n, len(image_entries))
    rows = (n + cols - 1) // cols
    indices = np.linspace(0, len(image_entries) - 1, num=n, dtype=int)

    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h))
    for slot, idx in enumerate(indices):
        entry = image_entries[idx]
        img = Image.open(entry) if isinstance(entry, str) else Image.fromarray(entry)
        img = img.convert("RGB").resize((thumb_w, thumb_h), Image.BILINEAR)
        r, c = divmod(slot, cols)
        sheet.paste(img, (c * thumb_w, r * thumb_h))

    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=jpeg_quality)
    return buf.getvalue()


def _export_video_from_entries(image_entries, out_path: Path, fps: int = 30) -> Path:
    """Encode an episode's frames into a standalone MP4 (used when use_video=True)."""
    import imageio.v2 as imageio

    if not image_entries:
        raise ValueError("Cannot export video: no frames in episode buffer.")

    with imageio.get_writer(str(out_path), fps=fps, codec="libx264") as w:
        for entry in image_entries:
            arr = imageio.imread(entry) if isinstance(entry, str) else entry
            w.append_data(arr)
    return out_path


def _analyze_contact_sheet(jpeg_bytes: bytes, model: str) -> dict:
    """Send a grid JPEG to Gemini and return the parsed outcome JSON."""
    import json

    from google import genai
    from google.genai import types

    from analyze_rollout import PROMPT, RESPONSE_SCHEMA

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.")

    client = genai.Client(api_key=api_key)
    image_part = types.Part(
        inline_data=types.Blob(mime_type="image/jpeg", data=jpeg_bytes),
    )

    response = client.models.generate_content(
        model=model,
        contents=[image_part, GRID_PROMPT_PREFIX + PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.1,
        ),
    )
    return json.loads(response.text)


def _analyze_with_retry(call, max_attempts: int = 10,
                        base_delay: float = 2.0, max_delay: float = 60.0) -> dict:
    """Call a zero-arg analysis function with exponential backoff. Raises on exhaustion."""
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return call()
        except Exception as e:
            last_err = e
            if attempt == max_attempts:
                break
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            print(f"  Gemini attempt {attempt}/{max_attempts} failed: {e}")
            print(f"  Retrying in {delay:.0f}s...")
            time.sleep(delay)
    raise RuntimeError(
        f"Gemini analysis failed after {max_attempts} attempts. Last error: {last_err}"
    )


def do_student_teacher_failure_analysis(
    student_policy_path,
    teacher_policy_path,
    repo_id_student=None,
    repo_id_teacher=None,
    main_task: str = "drop cube in the target region",
    num_student_episodes: int = 20,
    episode_time_s: int = 60,
    gemini_model: str = "gemini-2.5-pro",
    use_video: bool = False,
    gemini_video_fps: float = 1.0,
    n_grid_frames: int = 16,
    grid_cols: int = 4,
    hold_between_student_and_teacher: bool = True,
):
    """
    Run a student/teacher failure-analysis loop driven by Gemini classification.

    Flow per iteration:
      go_home → student rollout → analyze with Gemini → save episode →
      if failure: (optionally hold position) → teacher rollout with the
      task prompt for that failure mode → save episode.

    Args:
        student_policy_path / teacher_policy_path: HF model IDs or local paths.
        repo_id_student / repo_id_teacher: dataset repos. Auto-derived from
                                            policy paths if not given.
        main_task: prompt the student gets each rollout.
        num_student_episodes: how many student rollouts to run.
        episode_time_s: max seconds per episode.
        gemini_model: Gemini model ID for outcome classification.
        use_video: if True, send a full MP4 to Gemini instead of a contact sheet.
        gemini_video_fps: only used when use_video=True (Gemini's frame sampling rate).
        n_grid_frames / grid_cols: contact-sheet shape.
        hold_between_student_and_teacher: if True, teacher resumes from the
            student's final pose; if False, arm goes home between them.
    """
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.pipeline_features import (
        aggregate_pipeline_dataset_features, create_initial_features,
    )
    from lerobot.datasets.utils import combine_feature_dicts
    from lerobot.datasets.video_utils import VideoEncodingManager
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.processor import make_default_processors
    from lerobot.processor.rename_processor import rename_stats
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.scripts.lerobot_record import record_loop
    from lerobot.utils.control_utils import (
        init_keyboard_listener, sanity_check_dataset_name,
        sanity_check_dataset_robot_compatibility,
    )
    from lerobot.utils.utils import init_logging
    from lerobot.utils.visualization_utils import init_rerun

    from analyze_rollout import analyze_video

    if repo_id_student is None:
        repo_id_student = repo_id_from_policy(student_policy_path)
    if repo_id_teacher is None:
        repo_id_teacher = repo_id_from_policy(teacher_policy_path)

    _check_starvation()

    if _rerun_is_running():
        print("  Rerun viewer already running on :9090, skipping restart.")
    else:
        # Clean up any stale rerun process before launching a fresh one.
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/f", "/im", "rerun.exe"], capture_output=True)
        else:
            subprocess.run(["pkill", "-f", "rerun"], capture_output=True)
        subprocess.Popen(["rerun", "--serve-web"])
        threading.Thread(target=_wait_and_open_viewer, daemon=True).start()

    init_logging()
    init_rerun(session_name="student_teacher")

    robot_cfg = SOFollowerRobotConfig(
        port="/dev/tty.usbmodem5AB01813041",
        id="student_arm",
        use_degrees=True,
        cameras={
            "camera1": OpenCVCameraConfig(
                index_or_path=0, fps=30, width=640, height=480, warmup_s=2,
            )
        },
    )
    robot = SOFollower(robot_cfg)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

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
        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.pretrained_path = policy_path
        cfg.device = "mps"
        # Cluster-trained models may have an absolute path baked into the VLM
        # backbone — override to the public HF model ID so it resolves locally.
        if hasattr(cfg, "vlm_model_name") and cfg.vlm_model_name.startswith("/"):
            cfg.vlm_model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        return cfg

    def _open_dataset(repo_id, policy_cfg):
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
                repo_id, fps=30, robot_type=robot.name,
                features=dataset_features, use_videos=True,
                image_writer_processes=0, image_writer_threads=4,
            )
        return ds

    print("\n--- Student policy ---")
    cfg_s = _load_cfg(student_policy_path)
    dataset_s = _open_dataset(repo_id_student, cfg_s)
    print("  Loading student policy weights...")
    policy_s = make_policy(cfg_s, ds_meta=dataset_s.meta)
    pre_s, post_s = make_pre_post_processors(
        policy_cfg=cfg_s,
        pretrained_path=student_policy_path,
        dataset_stats=rename_stats(dataset_s.meta.stats, {}),
        preprocessor_overrides={
            "device_processor": {"device": "mps"},
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    print("\n--- Teacher policy ---")
    cfg_t = _load_cfg(teacher_policy_path)
    dataset_t = _open_dataset(repo_id_teacher, cfg_t)
    print("  Loading teacher policy weights...")
    policy_t = make_policy(cfg_t, ds_meta=dataset_t.meta)
    pre_t, post_t = make_pre_post_processors(
        policy_cfg=cfg_t,
        pretrained_path=teacher_policy_path,
        dataset_stats=rename_stats(dataset_t.meta.stats, {}),
        preprocessor_overrides={
            "device_processor": {"device": "mps"},
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    robot.connect()
    listener, events = init_keyboard_listener()

    counts = {"student": 0, "teacher": 0,
              "success": 0, "fail_pickup": 0, "fail_move": 0, "fail_drop": 0,
              "unknown_outcome": 0}

    try:
        with VideoEncodingManager(dataset_s):
            with VideoEncodingManager(dataset_t):
                for i in range(num_student_episodes):
                    if events.get("stop_recording", False):
                        break

                    events["exit_early"] = False
                    events["rerecord_episode"] = False

                    _go_home_with_robot(robot)

                    print(f"\n=== Episode {i + 1}/{num_student_episodes} | STUDENT ===")
                    print(f"  Task: {main_task}")
                    record_loop(
                        robot=robot, events=events, fps=30,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        policy=policy_s, preprocessor=pre_s, postprocessor=post_s,
                        dataset=dataset_s, control_time_s=episode_time_s,
                        single_task=main_task, display_data=False,
                    )
                    counts["student"] += 1

                    if not _episode_has_frames(dataset_s):
                        print("  WARNING: empty student episode, skipping analysis & save.")
                        if dataset_s.episode_buffer is not None:
                            dataset_s.clear_episode_buffer()
                        continue

                    # Wait for the image writer so PNG paths in the buffer point
                    # at files that actually exist on disk.
                    iw = getattr(dataset_s, "image_writer", None)
                    if iw is not None and hasattr(iw, "wait_until_done"):
                        iw.wait_until_done()

                    image_entries = dataset_s.episode_buffer["observation.images.camera1"]

                    print("  Sending rollout to Gemini for failure analysis...")
                    if use_video:
                        with tempfile.TemporaryDirectory() as td:
                            tmp = Path(td) / "rollout.mp4"
                            _export_video_from_entries(image_entries, tmp, fps=30)
                            outcome_dict = _analyze_with_retry(
                                lambda: analyze_video(tmp, model=gemini_model, fps=gemini_video_fps)
                            )
                    else:
                        jpeg = _build_contact_sheet(image_entries,
                                                    n=n_grid_frames, cols=grid_cols)
                        outcome_dict = _analyze_with_retry(
                            lambda: _analyze_contact_sheet(jpeg, model=gemini_model)
                        )

                    outcome    = outcome_dict.get("outcome", "unknown")
                    confidence = outcome_dict.get("confidence", 0.0)
                    reasoning  = outcome_dict.get("reasoning", "")
                    print(f"  → outcome:    {outcome}  (confidence {confidence:.2f})")
                    print(f"  → reasoning:  {reasoning}")

                    # Save the student episode AFTER analysis (analysis reads
                    # from episode_buffer, which save_episode() clears).
                    dataset_s.save_episode()

                    if outcome == "success":
                        counts["success"] += 1
                        continue

                    task_prompt = OUTCOME_TO_TASK.get(outcome)
                    if task_prompt is None:
                        print(f"  WARNING: unrecognized outcome '{outcome}', skipping teacher.")
                        counts["unknown_outcome"] += 1
                        continue

                    if   outcome == "failure_to_pick_up_block":                  counts["fail_pickup"] += 1
                    elif outcome == "failure_to_move_block_over_correct_region": counts["fail_move"]   += 1
                    elif outcome == "failure_to_drop_block":                     counts["fail_drop"]   += 1

                    if not hold_between_student_and_teacher:
                        _go_home_with_robot(robot)

                    print(f"\n--- Episode {i + 1}/{num_student_episodes} | TEACHER ---")
                    print(f"  Task: {task_prompt}")
                    events["exit_early"] = False
                    events["rerecord_episode"] = False
                    record_loop(
                        robot=robot, events=events, fps=30,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        policy=policy_t, preprocessor=pre_t, postprocessor=post_t,
                        dataset=dataset_t, control_time_s=episode_time_s,
                        single_task=task_prompt, display_data=False,
                    )
                    counts["teacher"] += 1

                    if _episode_has_frames(dataset_t):
                        dataset_t.save_episode()
                    elif dataset_t.episode_buffer is not None:
                        dataset_t.clear_episode_buffer()

                    if events.get("stop_recording", False):
                        break

    finally:
        _go_home_with_robot(robot)
        robot.disconnect()
        listener.stop()
        dataset_s.finalize()
        dataset_t.finalize()
        print("\n=== Run summary ===")
        for k, v in counts.items():
            print(f"  {k:>18}: {v}")
