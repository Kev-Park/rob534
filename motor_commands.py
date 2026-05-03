"""
Simple motor command utilities for the SO-101 arm.

Run directly from terminal:
    python motor_commands.py go_home
    python motor_commands.py get_pos
    python motor_commands.py reset_home
"""

import json
import sys
import time
from pathlib import Path

PORT         = "COM5"
CAMERA_INDEX = 1

_HOME_FILE = Path(__file__).parent / "home_pos.json"

_DEFAULT_HOME = {
    "shoulder_pan.pos":    3.6,
    "shoulder_lift.pos":  -92.4,
    "elbow_flex.pos":     97.5,
    "wrist_flex.pos":     76.7,
    "wrist_roll.pos":    -87.6,
    "gripper.pos":         3.2,
}


def load_home() -> dict:
    """Return saved home position, or defaults if none saved yet."""
    if _HOME_FILE.exists():
        return json.loads(_HOME_FILE.read_text())
    return _DEFAULT_HOME.copy()


def _connect(port=PORT):
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    config = SOFollowerRobotConfig(port=port, id="student_arm", use_degrees=True)
    robot = SOFollower(config)
    robot.connect(calibrate=False)
    return robot


def go_home(steps: int = 30, step_delay: float = 0.1, port: str = PORT):
    """Move arm smoothly to the saved home position."""
    home = load_home()
    robot = _connect(port)
    obs = robot.get_observation()
    current = {k: v for k, v in obs.items() if k.endswith(".pos")}
    print(f"Moving to home ({steps} steps × {step_delay}s)...")
    for i in range(1, steps + 1):
        t = i / steps
        interp = {k: current[k] + t * (home[k] - current[k]) for k in home}
        robot.send_action(interp)
        time.sleep(step_delay)
    robot.disconnect()
    print("Done.")


def get_pos(port: str = PORT) -> dict:
    """Print current joint positions."""
    robot = _connect(port)
    obs = robot.get_observation()
    robot.disconnect()
    pos = {k: round(float(v), 1) for k, v in obs.items() if k.endswith(".pos")}
    print("\nCurrent positions:")
    for k, v in pos.items():
        print(f"  {k}: {v}")
    return pos


def reset_home(port: str = PORT) -> dict:
    """Save current joint positions as the new home position."""
    robot = _connect(port)
    obs = robot.get_observation()
    robot.disconnect()
    pos = {k: round(float(v), 1) for k, v in obs.items() if k.endswith(".pos")}
    _HOME_FILE.write_text(json.dumps(pos, indent=2))
    print(f"Home saved to {_HOME_FILE}:")
    for k, v in pos.items():
        print(f"  {k}: {v}")
    return pos


def release(port: str = PORT):
    """Disable torque on all motors so the arm can be moved freely by hand."""
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    config = SOFollowerRobotConfig(port=port, id="student_arm", use_degrees=True,
                                   disable_torque_on_disconnect=True)
    robot = SOFollower(config)
    robot.connect(calibrate=False)
    robot.disconnect()  # disable_torque_on_disconnect=True releases all motors
    print("Motors released — arm is free to move.")


_COMMANDS = {
    "go_home":    go_home,
    "get_pos":    get_pos,
    "reset_home": reset_home,
    "release":    release,
}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in _COMMANDS:
        _COMMANDS[sys.argv[1]]()
    else:
        print(f"Usage: python motor_commands.py [{' | '.join(_COMMANDS)}]")
