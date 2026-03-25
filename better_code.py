import robot_control as rc

REPO_IDS = {
    "skywalker": "SkywalkerLi/so101_03_21_26_demo1",
    "nicole": "nc8304/so101_032326_white_back"
}

NICOLE_ROBOT = {**rc.ROBOT_DEFAULTS, "port": "COM5", "cameras": "{front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}"}
NICOLE_TELEOP = {**rc.TELEOP_DEFAULTS, "port": "COM4"}

def do_teleoperate():
    rc.teleoperate()


def do_record(repo_id="nc8304/so101_v2", num_episodes=5, single_task="Testing", resume=False):
    if repo_id == REPO_IDS["nicole"]:
        rc.record(repo_id=repo_id, num_episodes=num_episodes, single_task=single_task, resume=resume,
                  robot=NICOLE_ROBOT, teleop=NICOLE_TELEOP)
    else:
        rc.record(repo_id=repo_id, num_episodes=num_episodes, single_task=single_task, resume=resume)

def do_replay(repo_id="nc8304/so101", episode=0):
    rc.replay(repo_id=repo_id, episode=episode)


if __name__ == "__main__":
    #do_teleoperate()
    do_record(repo_id=REPO_IDS["nicole"], num_episodes=20, single_task="Nicole Data ", resume=False) #if file exsists make new one
    #do_replay(repo_id="nc8304/so101_031626",episode=0)
