import subprocess


# ── Shared defaults ────────────────────────────────────────────────────────────

ROBOT_DEFAULTS = {
    "type":    "so101_follower",
    "port":    "/dev/tty.usbmodem5AB01813041", #"COM5",
    "id":      "student_arm",
    "cameras": "{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, warmup_s: 10}}",
}

# SmolVLA expects `observation.images.camera1`, so we name the camera directly `camera1`.
SMOLVLA_ROBOT_DEFAULTS = {
    **ROBOT_DEFAULTS,
    "cameras": "{camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, warmup_s: 10}}",
}

TELEOP_DEFAULTS = {
    "type": "so101_leader",
    "port": "/dev/tty.usbmodem5A7C1216851", #"COM4",
    "id":   "teacher_arm",
}


def _run(cmd: list[str]) -> None:
    """Run a shell command, streaming output to the terminal."""
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ── Public functions ───────────────────────────────────────────────────────────

def teleoperate(
    robot=ROBOT_DEFAULTS,
    teleop=TELEOP_DEFAULTS,
):
    """
    Start live teleoperation: leader arm controls the follower arm in real time.

    Example:
        teleoperate()
    """
    _run([
        "lerobot-teleoperate",
        f"--robot.type={robot['type']}",
        f"--robot.port={robot['port']}",
        f"--robot.id={robot['id']}",
        f"--teleop.type={teleop['type']}",
        f"--teleop.port={teleop['port']}",
        f"--teleop.id={teleop['id']}",
        f"--robot.cameras={robot['cameras']}",
        f"--display_data=true"
    ])


def record(
    repo_id:            str  = "nc8304/so101_test2",
    num_episodes:       int  = 5,
    single_task:        str  = "Testing",
    display_data:       bool = True,
    streaming_encoding: bool = True,
    encoder_threads:    int  = 2,
    resume:             bool = False,
    robot=ROBOT_DEFAULTS,
    teleop=TELEOP_DEFAULTS,
):
    """
    Record demonstration episodes and push them to a Hugging Face dataset.

    Set resume=True to append episodes to an existing dataset.

    Example:
        record(repo_id="nc8304/so101", num_episodes=10, single_task="Pick and place")
        record(repo_id="nc8304/so101", resume=True)
    """
    cmd = [
        "lerobot-record",
        f"--robot.type={robot['type']}",
        f"--robot.port={robot['port']}",
        f"--robot.id={robot['id']}",
        f"--teleop.type={teleop['type']}",
        f"--teleop.port={teleop['port']}",
        f"--teleop.id={teleop['id']}",
        f"--robot.cameras={robot['cameras']}",
        f"--display_data={str(display_data).lower()}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.single_task={single_task}",
        f"--dataset.streaming_encoding={str(streaming_encoding).lower()}",
        f"--dataset.encoder_threads={encoder_threads}",
    ]

    if resume:
        cmd.append("--resume=true")

    _run(cmd)


def eval(
    policy_path:        str,
    repo_id:            str  = "eval/so101_eval",
    num_episodes:       int  = 5,
    single_task:        str  = "Testing",
    display_data:       bool = True,
    streaming_encoding: bool = True,
    encoder_threads:    int  = 2,
    resume:             bool = False,
    robot=ROBOT_DEFAULTS,
):
    """
    Run a trained policy on the robot and record evaluation episodes locally
    under an 'eval/' folder.

    Example:
        eval(policy_path="outputs/train/my_policy/checkpoints/last/pretrained_model",
             repo_id="eval/so101_eval", num_episodes=10)
    """
    cmd = [
        "lerobot-record",
        f"--robot.type={robot['type']}",
        f"--robot.port={robot['port']}",
        f"--robot.id={robot['id']}",
        f"--robot.cameras={robot['cameras']}",
        f"--display_data={str(display_data).lower()}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root=eval",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.single_task={single_task}",
        f"--dataset.streaming_encoding={str(streaming_encoding).lower()}",
        f"--dataset.encoder_threads={encoder_threads}",
        f"--policy.path={policy_path}",
    ]

    if resume:
        cmd.append("--resume=true")

    _run(cmd)


def smol_vla_eval(
    policy_path:        str,
    repo_id:            str  = "SkywalkerLi/eval_smolvla_cube",
    single_task:        str  = "Grab the cube",
    num_episodes:       int  = 10,
    episode_time_s:     int  = 60,
    reset_time_s:       int  = 10,
    policy_device:      str  = "mps",
    push_to_hub:        bool = False,
    display_data:       bool = True,
    resume:             bool = False,
    robot=SMOLVLA_ROBOT_DEFAULTS,
):
    """
    Run a SmolVLA policy on the robot.

    SmolVLA expects `observation.images.camera1`. We name the camera `camera1`
    at the robot config, so the dataset feature matches directly and no
    rename_map is required (lerobot 0.5.0's lerobot_record.py doesn't forward
    rename_map to make_policy, so the --dataset.rename_map flag does not help
    here).

    Example:
        smol_vla_eval(
            policy_path="SkywalkerLi/smol-vla-so101/pretrained_model",
        )
    """
    cmd = [
        "lerobot-record",
        f"--robot.type={robot['type']}",
        f"--robot.port={robot['port']}",
        f"--robot.id={robot['id']}",
        f"--robot.cameras={robot['cameras']}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.single_task={single_task}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.reset_time_s={reset_time_s}",
        f"--dataset.push_to_hub={str(push_to_hub).lower()}",
        f"--policy.path={policy_path}",
        f"--policy.device={policy_device}",
        f"--display_data={str(display_data).lower()}",
    ]

    if resume:
        cmd.append("--resume=true")
    _run(cmd)


def replay(
    repo_id:  str = "nc8304/so101",
    episode:  int = 0,
    robot=ROBOT_DEFAULTS,
):
    """
    Replay a recorded episode on the robot.

    Example:
        replay(repo_id="nc8304/so101", episode=2)
    """
    _run([
        "lerobot-replay",
        f"--robot.type={robot['type']}",
        f"--robot.port={robot['port']}",
        f"--robot.id={robot['id']}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.episode={episode}",
    ])


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Uncomment whichever you want to run:

    # teleoperate()
    # record(repo_id="nc8304/so101", num_episodes=5, single_task="Testing")
    # record(repo_id="nc8304/so101", num_episodes=5, single_task="Testing", resume=True)
    # replay(repo_id="nc8304/so101", episode=0)
    pass
