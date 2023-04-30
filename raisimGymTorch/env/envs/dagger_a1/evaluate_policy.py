from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import dagger_a1
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import os
import time
import numpy as np
import torch
import sys
import pandas as pd

VIZ = False
use_expert_gamma = False
use_expert_z = False
# Evaluating Params
num_episodes = 100

base_dir = sys.argv[1]
runid = sys.argv[2]

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(sys.argv[1] + "/cfg.yaml", 'r'))
# Multiple thread evaluation is not repeatable
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
cfg['environment']['test'] = False
cfg['environment']['eval'] = True
t_steps = cfg['environment']['history_len']

# create environment from the configuration file
env = VecEnv(dagger_a1.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts


prop_enc_pth = '/'.join([base_dir, 'prop_encoder_' + runid + '.pt'])
geom_enc_pth = '/'.join([base_dir, 'geom_encoder_' + runid + '.pt'])
mlp_pth = '/'.join([base_dir, 'mlp_' + runid + '.pt'])

env.load_scaling(base_dir, int(runid)) 

prop_loaded_encoder = torch.jit.load(prop_enc_pth)
loaded_mlp = torch.jit.load(mlp_pth)

if use_expert_gamma or use_expert_z:
    expert = torch.jit.load(os.path.join(base_dir, "policy_34000.pt"))

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
env.set_itr_number(int(runid))
ep = 0
k = 0
latent_p = None

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
        if k%2 == 0:
            if (latent_p is None) or (not use_expert_gamma or not use_expert_z):
                latent_p = prop_loaded_encoder(obs_torch[:,:base_dims*t_steps])
            if use_expert_gamma:
                expert_g = expert.geom_encoder(obs_torch[:,-5:-3])
                expert_g_2 = expert.geom_encoder(obs_torch[:,-3:-1])
                latent_p[:,-2] = expert_g[:,0]
                latent_p[:,-1] = expert_g_2[:,0]
            if use_expert_z:
                expert_p = expert.prop_encoder(obs_torch[:,-28:-5])
                latent_p[:,:8] = expert_p
        action_ll = loaded_mlp(torch.cat([obs_torch[:,base_dims*t_steps : base_dims*(t_steps + 1)],
                                              latent_p], 1))
        reward_ll, dones = env.step(action_ll.cpu().detach().numpy())

    eplen+=1
    metrics += np.array(env.get_reward_info())[:, metric_idxs]
    terminated = np.where(dones == 1)[0].tolist()
    k += 1
    for i in terminated:
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
