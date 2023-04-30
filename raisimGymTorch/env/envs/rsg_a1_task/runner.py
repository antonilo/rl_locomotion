from statistics import geometric_mean
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import rsg_a1_task
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse
try:
    import wandb
except:
    wandb = None

parser = argparse.ArgumentParser()
parser.add_argument("--exptid", type = int, help='experiment id to prepend to the run')
parser.add_argument("--overwrite", action = 'store_true')
parser.add_argument("--debug", action = 'store_true')
parser.add_argument("--loadid", type = int, default = None)
parser.add_argument("--gpu", type = int, default = 1)
parser.add_argument("--name", type = str)
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

rng_seed = cfg['seed']
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

activation_fn_map = {'none': None, 'tanh': nn.Tanh}
output_activation_fn = activation_fn_map[cfg['architecture']['activation']]
small_init_flag = cfg['architecture']['small_init']

if args.debug:
    cfg['environment']['num_envs'] = 1
    cfg['environment']['num_threads'] = 1
    device_type = 'cpu'
else:
    device_type = 'cuda:{}'.format(args.gpu)

cfg['environment']['test'] = False
cfg['environment']['speedTest'] = False

baseDim = cfg['environment']['baseDim']
priv_info = cfg['environment']['privinfo']
use_fourier = 'fourier' in cfg['environment'] and cfg['environment']['fourier']
geomDim = int(cfg['environment']['geomDim'])*int(cfg['environment']['use_slope_dots'])
n_futures = int(cfg['environment']['n_futures'])

# create environment from the configuration file
env = VecEnv(rsg_a1_task.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

if use_fourier:
    privy_dim = ob_dim - baseDim
    encoder_dim = cfg['environment']['encoder_dim']
    regular_fourier_dim = cfg['environment']['regular_fourier_dim']
    privy_fourier_dim = cfg['environment']['privy_fourier_dim']
    fourier_scale = cfg['environment']['fourier_scale']
    fourier_trainable = cfg['environment']['fourier_trainable']

    fourier_policy = cfg['environment']['fourier_policy']
    fourier_value = cfg['environment']['fourier_value']

# save the configuration and other files
saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/rsg_a1_task/" + '{:04d}'.format(args.exptid),
                           save_items=[task_path + "/Environment.hpp", task_path + "/runner.py"], config = cfg, overwrite = args.overwrite)
if wandb:
    wandb.init(project='command_loco', config=dict(cfg), name=args.name)
    wandb.save(home_path + '/raisimGymTorch/env/envs/rsg_a1_task/Environment.hpp')

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

if cfg['environment']['unnormalize_speed_vec']:
    raise NotImplementedError()

speed_vec_start_idx = cfg['architecture']['speed_vec_start_idx']
speed_vec_end_idx = cfg['architecture']['speed_vec_end_idx']
layer_type = cfg['architecture']['layer_type']
freeze_encoder = cfg['architecture']['freeze_encoder']

avg_rewards = []

if use_fourier:
    raise Exception('not implemented')
else:
    if priv_info:
        if layer_type == 'feedforward':
            init_var = 0.3
            module_type = ppo_module.MLPEncode_wrap
            actor = ppo_module.Actor(module_type(cfg['architecture']['policy_net'],
                                     nn.LeakyReLU,
                                     ob_dim//2,
                                     act_dim,
                                     output_activation_fn,
                                     small_init_flag,
                                     base_obdim = baseDim,
                                     geom_dim = geomDim,
                                     n_futures = n_futures),
                                     ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, init_var),
                                     device_type)

            critic = ppo_module.Critic(module_type(cfg['architecture']['value_net'],
                                                    nn.LeakyReLU,
                                                    ob_dim//2,
                                                    1,
                                                    base_obdim = baseDim,
                                                    geom_dim = geomDim,
                                                    n_futures = n_futures),
                                    device_type)
        else:
            raise NotImplementedError()

    else:
        raise NotImplementedError()

# Steps + flat policy
flat_policy_load_path = os.path.join(task_path,"../../../../data/base_policy/policy_22000.pt")
env.load_scaling(os.path.join(task_path, "../../../../data/base_policy"),
                 22000, policy_type=0, num_g1=n_futures)
loaded_graph_flat = torch.jit.load(flat_policy_load_path, map_location=torch.device(device_type))
flat_expert = ppo_module.Steps_Expert(loaded_graph_flat, device=device_type, baseDim=42,
                                      geomDim=2, n_futures=1, num_g1=n_futures)
# Encoders loading from blind stairs policy
checkpoint = torch.load(os.path.join(task_path,"../../../../data/base_policy/full_22000.pt"))
blind_policy_state_dict = checkpoint['actor_architecture_state_dict']
own_state = actor.architecture.state_dict()
for name, param in blind_policy_state_dict.items():
    own_state[name].copy_(param)
env.load_scaling(os.path.join(task_path, "../../../../data/base_policy"),
                 22000, policy_type=2, num_g1=n_futures)


ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.997,
              lam=0.95,
              num_mini_batches=4,
              device=device_type,
              log_dir=saver.data_dir,
              mini_batch_sampling='in_order',
              learning_rate=5e-4,
              flat_expert=flat_expert
              )

if wandb:
    wandb.watch(actor.architecture.architecture, log_freq=100)
    wandb.watch(critic.architecture.architecture, log_freq=100)

penalty_scale = np.array([cfg['environment']['lateralVelRewardCoeff'], cfg['environment']['angularVelRewardCoeff'], cfg['environment']['deltaTorqueRewardCoeff'], cfg['environment']['actionRewardCoeff'], cfg['environment']['sidewaysRewardCoeff'], cfg['environment']['jointSpeedRewardCoeff'], cfg['environment']['deltaContactRewardCoeff'], cfg['environment']['deltaReleaseRewardCoeff'], cfg['environment']['footSlipRewardCoeff'], cfg['environment']['upwardRewardCoeff'], cfg['environment']['workRewardCoeff'], cfg['environment']['yAccRewardCoeff'], 1., 1., 1.])

if args.loadid is not None:
    checkpoint = torch.load(saver.data_dir+"/full_"+str(args.loadid)+'.pt')
    actor.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
    actor.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
    critic.architecture.load_state_dict(checkpoint['critic_architecture_state_dict'])
    try:
        ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        print("Not loading ppo state")
    env.load_scaling(saver.data_dir, args.loadid, policy_type=1) 

if freeze_encoder:
    # do not update some networks
    print("Freezing the encoders!")
    for net_i in [actor.architecture.architecture.geom_encoder,
                  actor.architecture.architecture.prop_encoder]:
        for param in net_i.parameters():
            param.requires_grad = False

if args.loadid is not None:
    env.set_itr_number(args.loadid)


# This coefficient controls how much the policy is optimized with RL. Change to 1 for taking away demonstrations from a previous policy.
rl_coeff = 0.3
ppo.update_rl_coeff(rl_coeff)


for update in range(500001) if args.loadid is None else range(args.loadid + 1, 500001):
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    forwardX_sum = 0
    penalty_sum = 0
    done_sum = 0
    average_dones = 0.

    if update %  cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        actor.save_deterministic_graph(saver.data_dir+"/policy_"+str(update)+'.pt', torch.rand(1, ob_dim).cpu())
        if update %  (1 * cfg['environment']['eval_every_n']) == 0:
            torch.save({
                'actor_architecture_state_dict': actor.architecture.state_dict(),
                'actor_distribution_state_dict': actor.distribution.state_dict(),
                'critic_architecture_state_dict': critic.architecture.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
            }, saver.data_dir+"/full_"+str(update)+'.pt')

        parameters = np.zeros([0], dtype=np.float32)
        for param in actor.deterministic_parameters():
            parameters = np.concatenate([parameters, param.cpu().detach().numpy().flatten()], axis=0)
        np.savetxt(saver.data_dir+"/policy_"+str(update)+'.txt', parameters)
        loaded_graph = torch.jit.load(saver.data_dir+"/policy_"+str(update)+'.pt')

        env.reset()
        env.save_scaling(saver.data_dir, str(update))

    # actual training
    for step in range(n_steps):
        obs = env.observe(not freeze_encoder)
        action = ppo.observe(obs)
        reward, dones = env.step(action)
        unscaled_reward_info = env.get_reward_info()
        forwardX = unscaled_reward_info[:, 0]
        penalty = unscaled_reward_info[:, 1:]
        ppo.step(value_obs=obs, rews=reward, dones=dones, infos=[])
        done_sum = done_sum + sum(dones)
        reward_ll_sum = reward_ll_sum + sum(reward)
        forwardX_sum += np.sum(forwardX)
        penalty_sum += np.sum(penalty, axis=0)

    env.curriculum_callback()

    # take st step to get value obs
    obs = env.observe(not freeze_encoder)
    ppo.update(actor_obs=obs,
               value_obs=obs,
               log_this_iteration=update % 10 == 0,
               update=update)
    
    end = time.time()
    
    forwardX = forwardX_sum / total_steps
    forwardXReward = forwardX_sum * cfg['environment']['forwardVelRewardCoeff'] / total_steps

    forwardY, forwardZ, deltaTorque, action, sideways, jointSpeed, deltaContact, deltaRelease, footSlip, upward, work, yAcc, torqueSquare, stepHeight, walkedDist = penalty_sum / total_steps
    forwardYReward, forwardZReward, deltaTorqueReward, actionReward, sidewaysReward, jointSpeedReward, deltaContactReward, deltaReleaseReward, footSlipReward, upwardReward, workReward, yAccReward, torq, stepHeight, walkedDist = scaled_penalty = penalty_sum * penalty_scale / total_steps

    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    actor.distribution.enforce_minimum_std((torch.ones(12)*0.2).to(device_type))
    if wandb:
        wandb.log({'forwardX': forwardX, 
        'forwardX_reward': forwardXReward, 
        'forwardY': forwardY, 
        'forwardY_reward': forwardYReward, 
        'forwardZ': forwardZ, 
        'forwardZ_reward': forwardZReward, 
        'deltaTorque': deltaTorque, 
        'deltaTorque_reward': deltaTorqueReward, 
        'action': action, 
        'stepHeight': stepHeight,
        'action_reward': actionReward, 
        'sideways': sideways, 
        'sideways_reward': sidewaysReward, 
        'jointSpeed': jointSpeed, 
        'jointSpeed_reward': jointSpeedReward, 
        'deltaContact': deltaContact, 
        'deltaContact_reward': deltaContactReward, 
        'deltaRelease': deltaRelease, 
        'deltaRelease_reward': deltaReleaseReward, 
        'footSlip': footSlip, 
        'footSlip_reward': footSlipReward, 
        'upward': upward, 
        'upward_reward': upwardReward, 
        'work': work, 
        'work_reward': workReward, 
        'yAcc': yAcc, 
        'yAcc_reward': yAccReward,
        'torqueSquare': torqueSquare,
        'dones': average_dones,
        'walkedDist': walkedDist})

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("average forward reward: ", '{:0.10f}'.format(forwardXReward)))
    print('{:<40} {:>6}'.format("average penalty reward: ", ', '.join(['{:0.4f}'.format(r) for r in scaled_penalty])))
    print('{:<40} {:>6}'.format("average walked dist: ", '{:0.10f}'.format(scaled_penalty[-1])))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("lr: ", '{:.4e}'.format(ppo.optimizer.param_groups[0]["lr"])))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('std: ')
    print(np.exp(actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')
