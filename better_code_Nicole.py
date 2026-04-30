import re
import shutil
import subprocess
import threading
import time
import webbrowser
from pathlib import Path
import robot_control as rc

REPO_IDS = {
<<<<<<< HEAD
    "skywalker": "SkywalkerLi/so101_Nicole_Test",
    "nicole": "nc8304/so101_032326_white_back_041026_cube"}
=======
    "skywalker": "SkywalkerLi/so101_03_21_26_data_v1",
    "nicole": "nc8304/so101_031626"



NICOLE_ROBOT = {**rc.ROBOT_DEFAULTS, "port": "COM5", "cameras": "{front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}"}
NICOLE_TELEOP = {**rc.TELEOP_DEFAULTS, "port": "COM4"}

def _resolve_repo_id(repo_id: str) -> str:
    """If local HF cache is empty, delete and reuse. If non-empty, increment trailing number."""
    cache_base = Path.home() / ".cache" / "huggingface" / "lerobot"
    while True:
        owner, name = repo_id.split("/", 1)
        cache_path = cache_base / owner / name
        if not cache_path.exists():
            return repo_id
        if not any(cache_path.iterdir()):
            shutil.rmtree(cache_path)
            print(f"Deleted empty cache: {cache_path}")
            return repo_id
        match = re.search(r'(\d+)$', name)
        if match:
            num = int(match.group()) + 1
            name = name[:match.start()] + str(num).zfill(len(match.group()))
            repo_id = f"{owner}/{name}"
        else:
            repo_id = repo_id + "_2"
        print(f"Cache non-empty, trying: {repo_id}")


def do_teleoperate(repo_id=REPO_IDS["nicole"]):
    rc.teleoperate()


def _wait_and_open_viewer(port=9090, timeout=30):
    import socket
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                break
        except OSError:
            time.sleep(0.5)
    webbrowser.open(f"http://localhost:{port}/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A9876%2Fproxy")

def do_record(repo_id="nc8304/so101_v2", num_episodes=5, single_task="Testing", resume=False):
    subprocess.Popen(["rerun", "--serve-web"])
    threading.Thread(target=_wait_and_open_viewer, daemon=True).start()
    use_nicole_hw = (repo_id == REPO_IDS["nicole"])
    resolved = _resolve_repo_id(repo_id)
    print(f"Recording to: {resolved}")
    if use_nicole_hw:
        rc.record(repo_id=resolved, num_episodes=num_episodes, single_task=single_task, resume=resume,
                  robot=NICOLE_ROBOT, teleop=NICOLE_TELEOP, streaming_encoding=False, display_data=True)
    else:
        rc.record(repo_id=resolved, num_episodes=num_episodes, single_task=single_task, resume=resume,
                  streaming_encoding=False, display_data=True)

def do_eval(policy_path, repo_id="SkywalkerLi/eval_so101", num_episodes=5, single_task="Testing", resume=False):
    rc.eval(policy_path=policy_path, repo_id=repo_id, num_episodes=num_episodes, single_task=single_task, resume=resume)

def do_smol_vla_eval(
    policy_path,
    repo_id="SkywalkerLi/eval_smolvla_cube",
    single_task="Grab the cube",
    num_episodes=1,
):
    rc.smol_vla_eval(
        policy_path=policy_path,
        repo_id=repo_id,
        single_task=single_task,
        num_episodes=num_episodes,
    )

def do_replay(repo_id="nc8304/so101", episode=0):
    rc.replay(repo_id=repo_id, episode=episode)

import psutil, gc, time

def relieve_cpu_pressure():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)  # lower priority
    gc.collect()               # free memory to reduce GC CPU overhead
    time.sleep(0)              # yield current time slice

if __name__ == "__main__":
<<<<<<< HEAD

    relieve_cpu_pressure()
    #do_teleoperate(repo_id=REPO_IDS["nicole"])
    do_record(repo_id=REPO_IDS["nicole"], num_episodes=50, single_task="Nicole Cube", resume=False) #f file exsists make new one
=======
    #do_teleoperate()
    #do_record(repo_id=REPO_IDS["skywalker"], num_episodes=10, single_task="Grab orange triangle", resume=True) #if file exsists make new one
>>>>>>> 14e63057c1ef513d907434559b0a9d1fbc777f90
    #do_replay(repo_id="nc8304/so101_031626",episode=0)
    #do_eval(policy_path="SkywalkerLi/act-so101")
    do_smol_vla_eval(policy_path="SkywalkerLi/smol_vla")
