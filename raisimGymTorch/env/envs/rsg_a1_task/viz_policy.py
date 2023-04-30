from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_a1_task
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import os
import math
import time
import numpy as np
import torch
import sys
np.set_printoptions(suppress=True, precision=3)

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."
base_dir = sys.argv[1]
runid = sys.argv[2]

# config
cfg = YAML().load(open(sys.argv[1] + "/cfg.yaml", 'r'))
cfg['environment']['num_envs'] = 1
cfg['environment']['num_threads'] = 1
cfg['environment']['render'] = True

cfg['environment']['test'] = True
# Uncomment this for more controlled tests
#cfg['environment']['randomize_friction'] = False
#cfg['environment']['randomize_mass'] = False
#cfg['environment']['randomize_motor_strength'] = False
#cfg['environment']['randomize_gains'] = False
#cfg['environment']['speedTest'] = False

# create environment from the configuration file
env = VecEnv(rsg_a1_task.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
policy_load_path = '/'.join([base_dir, 'policy_' + runid + '.pt'])
env.load_scaling(base_dir, int(runid))
loaded_graph = torch.jit.load(policy_load_path)


foot_contacts = []
eplen = 0

print("Visualizing and evaluating the current policy")
for update in range(1):
    env.reset()
    env.turn_on_visualization()

    eplen = 0
    # An high number, assumes curriculum is finished
    env.set_itr_number(30000) #int(runid))
    for step in range(50000):
        time.sleep(0.01)
        obs = env.observe(False)
        with torch.no_grad():
            action_ll = loaded_graph(torch.from_numpy(obs).cpu())
        action = action_ll.cpu().detach().numpy()

        reward_ll, dones = env.step(action)
        reward_info = env.get_reward_info()[0]
        eplen+=1
        if dones[0]:
            eplen = 0
            # You can put plotting stuff here!

    env.turn_off_visualization()
