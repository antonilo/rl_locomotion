from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import dagger_a1
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import os
import math
import time
import torch
import sys

base_dir = sys.argv[1]
runid = sys.argv[2]

draw_plots = False
use_expert_gamma = False

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(sys.argv[1] + "/cfg.yaml", 'r'))
cfg['environment']['num_envs'] = 1
cfg['environment']['num_threads'] = 1
cfg['environment']['render'] = True

#cfg['environment']['randomize_friction'] = False
#cfg['environment']['randomize_mass'] = False
#cfg['environment']['randomize_motor_strength'] = False
#cfg['environment']['randomize_gains'] = False
#cfg['environment']['speedTest'] = False

base_dims = cfg['environment']['baseDim']
n_futures = cfg['environment']['n_futures']
cfg['environment']['test'] = True
t_steps = cfg['environment']['history_len']

# create environment from the configuration file
env = VecEnv(dagger_a1.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts


# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])

prop_enc_pth = '/'.join([base_dir, 'prop_encoder_' + runid + '.pt'])
mlp_pth = '/'.join([base_dir, 'mlp_' + runid + '.pt'])

env.load_scaling(base_dir, int(runid))

prop_loaded_encoder = torch.jit.load(prop_enc_pth)
loaded_mlp = torch.jit.load(mlp_pth)



print("Visualizing and evaluating the current policy")
eplen = 0
for update in range(1):
    env.reset()
    env.turn_on_visualization()

    eplen = 0
    env.set_itr_number(int(runid))
    for step in range(50000):
        time.sleep(0.01)
        obs = env.observe(False)

        obs_torch = torch.from_numpy(obs).cpu()
        with torch.no_grad():
            if step%2 == 0:
                latent_p = prop_loaded_encoder(obs_torch[:,:base_dims*t_steps])
            action_ll = loaded_mlp(torch.cat([obs_torch[:,base_dims*t_steps : base_dims*(t_steps + 1)],
                                              latent_p], 1))
        reward_ll, dones = env.step(action_ll.cpu().detach().numpy())
        eplen+=1
        work = env.get_reward_info()[0][-1]
        if dones[0]:
            eplen = 0
            # you can add plots here

    env.turn_off_visualization()
