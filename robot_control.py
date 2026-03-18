import subprocess


# ── Shared defaults ────────────────────────────────────────────────────────────

ROBOT_DEFAULTS = {
    "type":    "so101_follower",
    "port":    "COM5",
    "id":      "student_arm",
    "cameras": "{front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}",
}

TELEOP_DEFAULTS = {
    "type": "so101_leader",
    "port": "COM4",
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