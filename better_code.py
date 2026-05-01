import robot_control as rc

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

def do_smol_vla_eval(
    policy_path,
    repo_id="SkywalkerLi/eval_smolvla_cube",
    single_task="pick up the cube and drop it over the target region",
    num_episodes=1,
    resume = False
):
    rc.smol_vla_eval(
        policy_path=policy_path,
        repo_id=repo_id,
        single_task=single_task,
        num_episodes=num_episodes,
        resume=resume
    )

def do_replay(repo_id="nc8304/so101", episode=0):
    rc.replay(repo_id=repo_id, episode=episode)


if __name__ == "__main__":
    #do_teleoperate()
    #do_record(repo_id=REPO_IDS["skywalker"], num_episodes=10, single_task="Grab orange triangle", resume=True) #if file exsists make new one
    #do_replay(repo_id="nc8304/so101_031626",episode=0)
    #do_eval(policy_path="SkywalkerLi/act-so101")
    do_smol_vla_eval(policy_path="models/smolvla-phase-split", repo_id="SkywalkerLi/eval_smolvla_phase_split", resume = True)
