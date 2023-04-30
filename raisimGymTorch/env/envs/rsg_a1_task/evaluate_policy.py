from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_a1_task
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import os
import time
import numpy as np
import torch
import sys
import pandas as pd

VIZ = False
# Evaluating Params
num_episodes = 100

base_dir = sys.argv[1]
runid = sys.argv[2]

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(sys.argv[1] + "/cfg.yaml", 'r'))
# Multiple thread evaluation
cfg['environment']['render'] = VIZ
cfg['environment']['num_envs'] = 1
cfg['environment']['num_threads'] = 1

#cfg['environment']['randomize_friction'] = False
#cfg['environment']['randomize_mass'] = False
#cfg['environment']['randomize_motor_strength'] = False
#cfg['environment']['randomize_gains'] = False
#cfg['environment']['speedTest'] = False

base_dims = cfg['environment']['baseDim']
n_futures = cfg['environment']['n_futures']
num_envs = cfg['environment']['num_envs']
cfg['environment']['test'] = VIZ
cfg['environment']['eval'] = True

# create environment from the configuration file
env = VecEnv(rsg_a1_task.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

# save the configuration and other files
# saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/roughTerrain",
#                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])


env.load_scaling(base_dir, int(runid)) 
expert = torch.jit.load(os.path.join(base_dir, "policy_{}.pt".format(runid)))

print("Visualizing and evaluating the current policy")
env.reset()
rng_seed = 100
env.seed(rng_seed)
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)
if VIZ:
    env.turn_on_visualization()

eplen = np.zeros(num_envs, dtype=int)
latent_v = None
env.set_itr_number(30000)
ep = 0
k = 0

metrics_name = ['num_steps', 'forward_r', 'distance', 'energy', 'smoothness', 'ground_impact']
metrics_list = []

# Forward Reward, Distance, Work, Smoothness, Ground Impact
metric_idxs = [0, 15, 11, 3, 7]
metrics = np.zeros((num_envs, len(metric_idxs)))

while ep < num_episodes:
    if VIZ:
        time.sleep(0.01)
    obs = env.observe(False)
    obs_torch = torch.from_numpy(obs).cpu()
    with torch.no_grad():
        action_ll = expert.forward(torch.from_numpy(obs).cpu())
        reward_ll, dones = env.step(action_ll.cpu().detach().numpy())

    eplen+=1
    metrics += np.array(env.get_reward_info())[:, metric_idxs]
    terminated = np.where(dones == 1)[0].tolist()
    k += 1
    for i in terminated:
        #print("Lenght is {}".format(eplen[i]))
        row = [int(eplen[i])]
        row.extend(metrics[i].tolist())
        if ep < num_episodes:
            metrics_list.append(row)
        ep += 1
        if ep % (num_episodes // 10) == 0:
            print("Done {}\%".format(ep / num_episodes*100))
        eplen[i] = 0
        metrics[i] *= 0
        k = 0
        if VIZ:
            env.turn_off_visualization()
            env.turn_on_visualization()
            time.sleep(0.5)
# Save as csv
path = os.path.join(sys.argv[1], "evaluation_results.csv")
pd.DataFrame(np.stack(metrics_list)).to_csv(path, header=metrics_name, index=None)
