import shutil
import socket
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

import robot_control as rc

# ── Home position ─────────────────────────────────────────────────────────────
# Joint angles in degrees the robot returns to between episodes.
# Adjust these values to match your desired rest/start position.
HOME_POS = {
    "shoulder_pan.pos":    3.6,
    "shoulder_lift.pos":  -92.4,
    "elbow_flex.pos":     97.5,
    "wrist_flex.pos":     76.7,
    "wrist_roll.pos":    -87.6,
    "gripper.pos":         3.2,
}


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
        home_pos = HOME_POS

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
            print(f"  Found local cache: {local_path}")
            return str(local_path)
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


def do_smol_vla_eval(
    policy_path,
    repo_id=None,
    single_task="Grab the cube and drop it ",
    num_episodes=1,
    reset_time_s=10,
):
    if repo_id is None:
        repo_id = repo_id_from_policy(policy_path)
    dataset_path = Path.home() / ".cache/huggingface/lerobot" / repo_id
    if dataset_path.exists():
        print(f"Removing existing dataset: {dataset_path}")
        shutil.rmtree(dataset_path)

    _check_starvation()
    subprocess.run(["taskkill", "/f", "/im", "rerun.exe"], capture_output=True)
    subprocess.Popen(["rerun", "--serve-web"])
    threading.Thread(target=_wait_and_open_viewer, daemon=True).start()

    for i in range(num_episodes):
        go_home()
        print(f"\n  Episode {i + 1}/{num_episodes}")
        rc.smol_vla_eval(
            policy_path=policy_path,
            repo_id=repo_id,
            single_task=single_task,
            num_episodes=1,
            reset_time_s=0,
            resume=(i > 0),
            display_data=True,
        )

    go_home()

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

    #do_teleoperate()
    #do_record(repo_id=REPO_IDS["skywalker"], num_episodes=10, single_task="Grab orange triangle", resume=True) #if file exsists make new one
    #do_replay(repo_id="nc8304/so101_031626",episode=0)
    #do_eval(policy_path="SkywalkerLi/act-so101")
    do_smol_vla_eval(policy_path=resolve_policy_path("SkywalkerLi/smolvla-phase-split"),num_episodes=10)
