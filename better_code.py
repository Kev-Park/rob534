import shutil
import socket
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np

import robot_control as rc
from motor_commands import load_home, PORT
from struggle_monitor import LiveStruggleMonitor


def _edge_removed_rgb(frame_rgb: np.ndarray, ksize: int = 3, threshold: int = 50) -> np.ndarray:
    """Match the training-time Sobel edge-removal applied by sobel_batch.py.

    Input: RGB uint8 HWC (lerobot OpenCVCamera default). Output: RGB uint8 HWC
    with pixels whose Sobel-magnitude >= threshold set to 0.
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.clip(np.sqrt(gx * gx + gy * gy), 0, 255).astype(np.uint8)
    out = frame_rgb.copy()
    out[mag >= threshold] = 0
    return out


def go_home(home_pos: dict = None, port: str = "COM5",
            steps: int = 30, step_delay: float = 0.1):
    """
    Smoothly move the arm to home_pos (defaults to HOME_POS).

    Movement is interpolated over `steps` steps with `step_delay` seconds
    between each — total duration = steps * step_delay (default ~1.5 s).

    Args:
        home_pos:   Target joint angles dict. Defaults to HOME_POS.
        port:       Serial port the arm is on.
        steps:      Number of interpolation steps (more = smoother/slower).
        step_delay: Seconds between steps (larger = slower).
    """
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    if home_pos is None:
        home_pos = load_home()

    config = SOFollowerRobotConfig(port=port, id="student_arm", use_degrees=True)
    robot = SOFollower(config)
    robot.connect(calibrate=False)

    # Read current position so we can interpolate from it
    obs = robot.get_observation()
    current = {k: v for k, v in obs.items() if k.endswith(".pos")}

    print(f"  Moving to home position ({steps} steps × {step_delay}s)...")
    for i in range(1, steps + 1):
        t = i / steps
        interp = {k: current[k] + t * (home_pos[k] - current[k]) for k in home_pos}
        robot.send_action(interp)
        time.sleep(step_delay)

    robot.disconnect()
    print("  Home position reached.")


def get_current_pos(port: str = "COM5"):
    """
    Connect to the arm, read joint positions, print them ready to paste into HOME_POS, then disconnect.
    """
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    config = SOFollowerRobotConfig(port=port, id="student_arm", use_degrees=True)
    robot = SOFollower(config)
    robot.connect(calibrate=False)
    obs = robot.get_observation()
    robot.disconnect()

    pos = {k: v for k, v in obs.items() if k.endswith(".pos")}
    print("\nCurrent joint positions (paste into HOME_POS):")
    print("HOME_POS = {")
    for k, v in pos.items():
        print(f'    "{k}": {v:.1f},')
    print("}")
    return pos


def _rerun_is_running(port=9090):
    try:
        with socket.create_connection(("localhost", port), timeout=0.5):
            return True
    except OSError:
        return False


def _wait_and_open_viewer(port=9090, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                break
        except OSError:
            time.sleep(0.5)
    webbrowser.open(f"http://localhost:{port}/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9876%2Fproxy")

REPO_IDS = {
    "skywalker": "SkywalkerLi/so101_03_21_26_data_v1",
    "nicole": "nc8304/so101_031626"
}

def do_teleoperate():
    rc.teleoperate()


def do_record(repo_id="nc8304/so101_v2", num_episodes=5, single_task="Testing", resume=True):
    rc.record(repo_id=repo_id, num_episodes=num_episodes, single_task=single_task, resume=resume)

def do_eval(policy_path, repo_id="SkywalkerLi/eval_so101", num_episodes=5, single_task="Testing", resume=False):
    rc.eval(policy_path=policy_path, repo_id=repo_id, num_episodes=num_episodes, single_task=single_task, resume=resume)

def _check_starvation():
    import psutil
    print("Checking for CPU starvation before opening camera...")
    # Sample CPU over 2 seconds for a stable reading
    cpu = psutil.cpu_percent(interval=2)
    ram = psutil.virtual_memory()
    ram_free_gb = (ram.total - ram.used) / 1024**3
    print(f"  CPU usage:  {cpu:.1f}%")
    print(f"  RAM free:   {ram_free_gb:.1f} GB")
    if cpu > 70:
        print(f"  WARNING: CPU is at {cpu:.1f}% - starvation likely, record loop may run slow")
    if ram_free_gb < 1.0:
        print(f"  WARNING: Low RAM ({ram_free_gb:.1f} GB free) - may cause slowness")
    if cpu <= 70 and ram_free_gb >= 1.0:
        print("  OK: System looks healthy")
    print("  NOTE: During SmolVLA eval, lerobot will log a '4 Hz' warning once per action chunk")
    print("        (~every 2s). This is expected — VLA inference takes ~238ms but only runs")
    print("        every 50 steps; the robot executes at ~26 Hz between calls.")


def resolve_policy_path(hf_model_id: str) -> str:
    """
    Given a HuggingFace model ID (e.g. 'SkywalkerLi/smolvla-phase-split'),
    return the local snapshot path if already cached, otherwise return the
    HF model ID so lerobot can download it.
    """
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    namespace, model_name = hf_model_id.split("/", 1)
    cache_dir = hf_cache / f"models--{namespace}--{model_name}"
    snapshots_dir = cache_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(snapshots_dir.iterdir())
        if snapshots:
            local_path = snapshots[-1]  # most recent snapshot
            # Only use local cache if weights are actually present (not just config.json)
            if any(local_path.glob("*.safetensors")):
                print(f"  Found local cache: {local_path}")
                return str(local_path)
            else:
                print(f"  Cache snapshot has no weights, will download: {hf_model_id}")
    print(f"  Not cached locally, will download: {hf_model_id}")
    return hf_model_id


def repo_id_from_policy(policy_path: str) -> str:
    """Derive a local/hub dataset repo_id from the policy path.

    SkywalkerLi/smolvla-phase-split            -> SkywalkerLi/eval_smolvla-phase-split
    .../models--SkywalkerLi--smolvla-phase-split/snapshots/abc  -> same
    """
    for part in Path(policy_path).parts:
        if part.startswith("models--"):
            _, namespace, model_name = part.split("--", 2)
            return f"{namespace}/eval_{model_name}"
    if "/" in policy_path:
        namespace, model_name = policy_path.split("/", 1)
        return f"{namespace}/eval_{model_name}"
    return f"eval_{policy_path}"


def _go_home_with_robot(robot, steps: int = 30, step_delay: float = 0.1):
    """Move to saved home position using an already-connected robot (no serial reconnect)."""
    home = load_home()
    obs = robot.get_observation()
    current = {k: v for k, v in obs.items() if k.endswith(".pos")}
    print(f"  Moving to home position ({steps} steps × {step_delay}s)...")
    for i in range(1, steps + 1):
        t = i / steps
        interp = {k: current[k] + t * (home[k] - current[k]) for k in home}
        robot.send_action(interp)
        time.sleep(step_delay)
    print("  Home position reached.")


def do_smol_vla_eval(
    policy_path,
    repo_id=None,
    single_task="Grab the cube and drop it ",
    num_episodes=1,
    episode_time_s=60,
    use_struggle_monitor=False,
    struggle_key_file=r"C:\Users\calle\Desktop\gem.txt",
    struggle_check_interval=2.0,
    struggle_threshold=0.6,
    use_edge_removed=False,
    edge_ksize=3,
    edge_threshold=50,
):
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
    from lerobot.utils.utils import init_logging, log_say
    from lerobot.utils.visualization_utils import init_rerun

    if repo_id is None:
        repo_id = repo_id_from_policy(policy_path)
    dataset_path = Path.home() / ".cache/huggingface/lerobot" / repo_id
    # Only resume if the dataset is complete (has actual data, not just a partial/failed folder)
    resuming = (dataset_path / "meta" / "tasks.parquet").exists()
    if resuming:
        print(f"  Found existing dataset at {dataset_path} — resuming (appending episodes).")
    elif dataset_path.exists():
        print(f"  Found incomplete dataset at {dataset_path} — starting fresh.")
        shutil.rmtree(dataset_path)

    _check_starvation()
    if _rerun_is_running():
        print("  Rerun viewer already running on :9090, skipping restart.")
    else:
        subprocess.run(["taskkill", "/f", "/im", "rerun.exe"], capture_output=True)
        subprocess.Popen(["rerun", "--serve-web"])
        threading.Thread(target=_wait_and_open_viewer, daemon=True).start()

    init_logging()
    init_rerun(session_name="recording")

    # ── One-time setup ────────────────────────────────────────────────────────
    # Everything below is created ONCE and reused across all episodes.
    # Previously we called lerobot-record as a subprocess per episode, which
    # meant reloading weights (~20-30s) and re-warming the camera (10s) every
    # single episode. Now we call lerobot's Python API directly so the robot
    # connection, camera, and policy weights stay in memory for the full run.

    # Robot + camera: opened once, camera warms up once (warmup_s=2).
    # When use_edge_removed is on, the policy was trained on observation.images.front,
    # so we name the camera "front" to match — same hardware (index_or_path=1).
    camera_key = "front" if use_edge_removed else "camera1"
    robot_cfg = SOFollowerRobotConfig(
        port="COM5",
        id="student_arm",
        use_degrees=True,
        cameras={
            camera_key: OpenCVCameraConfig(
                index_or_path=1, fps=30, width=640, height=480, warmup_s=2,
            )
        },
    )
    robot = SOFollower(robot_cfg)

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = "cuda"
    # The policy was trained on a cluster whose local path is baked into config.json.
    # Override it to the public HF model ID so transformers can find it locally.
    if hasattr(policy_cfg, "vlm_model_name") and policy_cfg.vlm_model_name.startswith("/"):
        policy_cfg.vlm_model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

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

    if resuming:
        # Load existing dataset and append to it.
        dataset = LeRobotDataset(repo_id, root=dataset_path)
        dataset.start_image_writer(num_processes=0, num_threads=4)
        sanity_check_dataset_robot_compatibility(dataset, robot, 30, dataset_features)
        print(f"  Resuming from episode {dataset.num_episodes} ({dataset.num_frames} frames so far).")
    else:
        # Fresh dataset — created for the first time.
        sanity_check_dataset_name(repo_id, policy_cfg)
        dataset = LeRobotDataset.create(
            repo_id,
            fps=30,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4,
        )

    # Policy weights loaded once here — SmolVLA-500M takes ~20-30s to load.
    # All episodes share the same policy object in GPU memory.
    print("  Loading policy weights (once for all episodes)...")
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        dataset_stats=rename_stats(dataset.meta.stats, {}),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    # Connect robot and camera once — stays open for all episodes.
    robot.connect()

    if use_edge_removed:
        _orig_get_obs = robot.get_observation
        def _get_obs_edge_removed():
            obs = _orig_get_obs()
            frame = obs.get(camera_key)
            if frame is not None:
                obs[camera_key] = _edge_removed_rgb(frame, edge_ksize, edge_threshold)
            return obs
        robot.get_observation = _get_obs_edge_removed
        print(f"  [edge-removed] applying Sobel(ksize={edge_ksize}, thr={edge_threshold}) "
              f"to '{camera_key}' frames live.")

    listener, events = init_keyboard_listener()

    # ── Struggle monitor (optional) ───────────────────────────────────────────
    monitor = None
    if use_struggle_monitor:
        monitor = LiveStruggleMonitor(
            key_file=struggle_key_file,
            model="gemini-2.5-flash",
            check_interval=struggle_check_interval,
            struggle_threshold=struggle_threshold,
        )
        monitor.start()
        monitor.start_capture(camera_index=1)
        print("  [StruggleMonitor] watching camera — will flag struggling episodes")

    def _struggle_watcher(stop_evt):
        """Background thread: sets exit_early when monitor signals struggling."""
        while not stop_evt.is_set():
            if monitor and monitor.is_struggling():
                sig = monitor.get_signal()
                print(f"\n  [StruggleMonitor] STRUGGLING — exiting episode early")
                print(f"  Reason: {sig['reason']}  (conf={sig['confidence']:.2f})")
                events["exit_early"] = True
                break
            stop_evt.wait(timeout=0.25)

    try:
        with VideoEncodingManager(dataset):
            for i in range(num_episodes):
                # Reset events so any stray keypress during go_home doesn't
                # immediately exit the first control loop iteration.
                events["exit_early"] = False
                events["rerecord_episode"] = False

                if monitor:
                    monitor.reset_signal()

                # Watcher thread for this episode
                watcher_stop = threading.Event()
                if monitor:
                    watcher = threading.Thread(
                        target=_struggle_watcher, args=(watcher_stop,), daemon=True
                    )
                    watcher.start()

                # Move arm to home position before each episode so every
                # episode starts from the same known configuration.
                # home_pos is loaded from home_pos.json (set via motor_commands.py reset_home).
                _go_home_with_robot(robot)
                print(f"\n  Episode {i + 1}/{num_episodes}")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=30,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=episode_time_s,
                    single_task=single_task,
                    display_data=True,
                )

                # Stop the per-episode watcher
                if monitor:
                    watcher_stop.set()

                frames_collected = (
                    dataset.episode_buffer is not None
                    and dataset.episode_buffer.get("size", 0) > 0
                )
                if frames_collected:
                    dataset.save_episode()
                else:
                    print(f"  WARNING: Episode {i + 1} collected no frames — skipping save.")
                    if dataset.episode_buffer is not None:
                        dataset.clear_episode_buffer()

                if events["stop_recording"]:
                    break
    finally:
        if monitor:
            monitor.stop()
        # Move back to home before disconnecting so the arm parks cleanly.
        _go_home_with_robot(robot)
        robot.disconnect()
        listener.stop()
        dataset.finalize()

def do_replay(repo_id="nc8304/so101", episode=0):
    rc.replay(repo_id=repo_id, episode=episode)


if __name__ == "__main__":
    import torch
    import psutil

    print("=" * 50)
    print("SYSTEM CHECK")
    print("=" * 50)

    # GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:            {torch.cuda.get_device_name(0)}")
        print(f"PyTorch CUDA:   {torch.version.cuda}")
        mem = torch.cuda.mem_get_info(0)
        free_mb = mem[0] / 1024**2
        total_mb = mem[1] / 1024**2
        used_mb = total_mb - free_mb
        print(f"GPU memory:     {used_mb:.0f} MB used / {total_mb:.0f} MB total")
        if used_mb > total_mb * 0.8:
            print("WARNING: GPU memory is >80% used - another process may be hogging it")

        # GPU power / performance check
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,power.limit,clocks.gr,clocks.max.gr", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if smi.returncode == 0:
            parts = [p.strip() for p in smi.stdout.strip().split(",")]
            try:
                power_draw  = float(parts[0]) if parts[0] != "[N/A]" else None
                power_limit = float(parts[1]) if parts[1] != "[N/A]" else None
                clk_cur     = float(parts[2]) if parts[2] != "[N/A]" else None
                clk_max     = float(parts[3]) if parts[3] != "[N/A]" else None

                if power_draw is not None:
                    plimit_str = f" / {power_limit:.0f} W limit" if power_limit else ""
                    print(f"GPU power:      {power_draw:.1f} W{plimit_str} [idle - will spike during inference]")

                if clk_cur is not None and clk_max is not None:
                    pct = clk_cur / clk_max * 100
                    print(f"GPU clock:      {clk_cur:.0f} MHz / {clk_max:.0f} MHz ({pct:.0f}%) [idle - will ramp under load]")
            except (ValueError, IndexError):
                pass
    else:
        print("WARNING: No GPU detected - policy will run on CPU (very slow!)")

    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU usage:      {cpu_percent:.1f}%")
    if cpu_percent > 70:
        print("WARNING: CPU is busy - background tasks may be causing slowness")

    # RAM
    ram = psutil.virtual_memory()
    print(f"RAM:            {ram.used / 1024**3:.1f} GB used / {ram.total / 1024**3:.1f} GB total")

    # nvidia-smi for running GPU processes
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,name", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if smi.stdout.strip():
            print(f"GPU processes:  {smi.stdout.strip()}")
        else:
            print("GPU processes:  none")
    except FileNotFoundError:
        print("GPU processes:  nvidia-smi not found")

    print("=" * 50)

    #get_current_pos()
    #do_teleoperate()
    #do_record(repo_id=REPO_IDS["skywalker"], num_episodes=10, single_task="Grab orange triangle", resume=True) #if file exsists make new one
    #do_replay(repo_id="nc8304/so101_031626",episode=0)
    #do_eval(policy_path="SkywalkerLi/act-so101")
    do_smol_vla_eval(
        policy_path=resolve_policy_path("SkywalkerLi/smolvla-aug"),
        repo_id="SkywalkerLi/eval_smolvla-aug",
        num_episodes=10,
        episode_time_s=45,
        use_struggle_monitor=True,
        use_edge_removed=True,
    )
