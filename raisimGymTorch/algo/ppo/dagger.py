import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .storage import ObsStorage 


# computes and returns the latent from the expert
class DaggerExpert(nn.Module):
    def __init__(self, loadpth, runid, total_obs_size, T, base_obs_size, nenvs, geomDim = 4, n_futures = 3):
        super(DaggerExpert, self).__init__()
        path = '/'.join([loadpth, 'policy_' + runid + '.pt'])
        self.policy = torch.jit.load(path)
        self.geomDim = geomDim
        self.n_futures = n_futures
        mean_pth = loadpth + "/mean" + runid + ".csv"
        var_pth = loadpth + "/var" + runid + ".csv"
        obs_mean = np.loadtxt(mean_pth, dtype=np.float32)
        obs_var = np.loadtxt(var_pth, dtype=np.float32)
        # cut it
        obs_mean = obs_mean[:,obs_mean.shape[1]//2:]
        obs_var = obs_var[:,obs_var.shape[1]//2:]
        self.mean = self.get_tiled_scales(obs_mean, nenvs, total_obs_size, base_obs_size, T)
        self.var = self.get_tiled_scales(obs_var, nenvs, total_obs_size, base_obs_size, T)
        self.tail_size = total_obs_size - (T + 1) * base_obs_size

    def get_tiled_scales(self, invec, nenvs, total_obs_size, base_obs_size, T):
        outvec = np.zeros([nenvs, total_obs_size], dtype = np.float32)
        outvec[:, :base_obs_size * T] = np.tile(invec[0, :base_obs_size], [1, T])
        outvec[:, base_obs_size * T:] = invec[0]
        return outvec

    def forward(self, obs):
        obs = obs[:,-self.tail_size:]
        with torch.no_grad():
            prop_latent = self.policy.prop_encoder(obs[:, :-self.geomDim*(self.n_futures+1)-1]) # since there is also ref at the end
            geom_latents = []
            for i in reversed(range(self.n_futures+1)):
                start = -(i+1)*self.geomDim -1
                end = -i*self.geomDim -1
                if (end == 0):
                    end = None
                geom_latent = self.policy.geom_encoder(obs[:,start:end])
                geom_latents.append(geom_latent)
            geom_latents = torch.hstack(geom_latents)
            expert_latent = torch.cat((prop_latent, geom_latents), dim=1)
        return expert_latent

class DaggerAgent:
    def __init__(self, expert_policy,
                 prop_latent_encoder, student_mlp,
                 T, base_obs_size, device, n_futures=3):
        expert_policy.to(device)
        prop_latent_encoder.to(device)
        #geom_latent_encoder.to(device)
        student_mlp.to(device)
        self.expert_policy = expert_policy
        self.prop_latent_encoder = prop_latent_encoder
        #self.geom_latent_encoder = geom_latent_encoder
        self.student_mlp = student_mlp
        self.base_obs_size = base_obs_size
        self.T = T
        self.device = device
        self.mean = expert_policy.mean
        self.var = expert_policy.var
        self.n_futures = n_futures
        self.itr = 0
        self.current_prob = 0
        # copy expert weights for mlp policy
        self.student_mlp.architecture.load_state_dict(self.expert_policy.policy.action_mlp.state_dict())


        for net_i in [self.expert_policy.policy, self.student_mlp]:
            for param in net_i.parameters():
                param.requires_grad = False

    def set_itr(self, itr):
        self.itr = itr
        if (itr+1) % 100 == 0:
            self.current_prob += 0.1
            print(f"Probability set to {self.current_prob}")

    def get_history_encoding(self, obs):
        hlen = self.base_obs_size * self.T
        raw_obs = obs[:, : hlen]
        # Hack to add velocity
        #velocity = obs[:, self.velocity_idx] -> Velocity thing is not robust
        #raw_obs[:, -3:] = velocity
        prop_latent = self.prop_latent_encoder(raw_obs)
        #geom_latent = self.geom_latent_encoder(raw_obs)
        return prop_latent

    def evaluate(self, obs):
        hlen = self.base_obs_size * self.T
        obdim = self.base_obs_size
        prop_latent = self.get_history_encoding(obs)
        #expert_latent = self.get_expert_latent(obs)
        #expert_future_geoms = expert_latent[:,prop_latent.shape[1]+geom_latent.shape[1]:]
        # assume that nothing changed
        #geom_latents = []
        #for i in range(self.n_futures + 1):
        #    geom_latents.append(geom_latent)
        #geom_latents = torch.hstack((geom_latent, expert_future_geoms))
        #if np.random.random() < self.current_prob:
        #    # student action
        output = torch.cat([obs[:, hlen : hlen + obdim], prop_latent], 1)
        #else:
        #    # expert action
        #    output = torch.cat([obs[:, hlen : hlen + obdim], expert_latent], 1)
        output = self.student_mlp.architecture(output)
        return output

    def get_expert_action(self, obs):
        hlen = self.base_obs_size * self.T
        obdim = self.base_obs_size
        expert_latent = self.get_expert_latent(obs)
        output = torch.cat([obs[:, hlen : hlen + obdim], expert_latent], 1)
        #else:
        #    # expert action
        #    output = torch.cat([obs[:, hlen : hlen + obdim], expert_latent], 1)
        output = self.student_mlp.architecture(output)
        return output

    def get_student_action(self, obs):
        return self.evaluate(obs)

    def get_expert_latent(self, obs):
        with torch.no_grad():
            latent = self.expert_policy(obs).detach()
            return latent

    def save_deterministic_graph(self, fname_prop_encoder,
                                 fname_mlp, example_input, device='cpu'):
        hlen = self.base_obs_size * self.T

        prop_encoder_graph = torch.jit.trace(self.prop_latent_encoder.to(device), example_input[:, :hlen])
        torch.jit.save(prop_encoder_graph, fname_prop_encoder)

        #geom_encoder_graph = torch.jit.trace(self.geom_latent_encoder.to(device), example_input[:, :hlen])
        #torch.jit.save(geom_encoder_graph, fname_geom_encoder)

        mlp_graph = torch.jit.trace(self.student_mlp.architecture.to(device), example_input[:, hlen:])
        torch.jit.save(mlp_graph, fname_mlp)

        self.prop_latent_encoder.to(self.device)
        #self.geom_latent_encoder.to(self.device)
        self.student_mlp.to(self.device)

class DaggerTrainer:
    def __init__(self,
            actor,
            num_envs, 
            num_transitions_per_env,
            obs_shape, latent_shape,
            num_learning_epochs=4,
            num_mini_batches=4,
            device=None,
            learning_rate=5e-4):

        self.actor = actor
        self.storage = ObsStorage(num_envs, num_transitions_per_env, [obs_shape], [latent_shape], device)
        self.optimizer = optim.Adam([*self.actor.prop_latent_encoder.parameters()],
                                    lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.1)
        self.device = device
        self.itr = 0

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.loss_fn = nn.MSELoss()

    def observe(self, obs):
        with torch.no_grad():
            actions = self.actor.get_student_action(torch.from_numpy(obs).to(self.device))
            #actions = self.actor.get_expert_action(torch.from_numpy(obs).to(self.device))
        return actions.detach().cpu().numpy()

    def step(self, obs):
        expert_latent = self.actor.get_expert_latent(torch.from_numpy(obs).to(self.device))
        self.storage.add_obs(obs, expert_latent)

    def update(self):
        # Learning step
        mse_loss = self._train_step()
        self.storage.clear()
        return mse_loss

    def _train_step(self):
        self.itr += 1
        self.actor.set_itr(self.itr)
        for epoch in range(self.num_learning_epochs):
            # return loss in the last epoch
            prop_mse = 0
            geom_mse = 0
            loss_counter = 0
            for obs_batch, expert_action_batch in self.storage.mini_batch_generator_inorder(self.num_mini_batches):

                predicted_prop_latent = self.actor.get_history_encoding(obs_batch)
                loss_prop = self.loss_fn(predicted_prop_latent[:,:8], expert_action_batch[:,:8])
                loss_geom = self.loss_fn(predicted_prop_latent[:,8:],
                        expert_action_batch[:,8:])
                loss = loss_geom + loss_prop

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                prop_mse += loss_prop.item()
                geom_mse += loss_geom.item()
                loss_counter += 1

            avg_prop_loss = prop_mse / loss_counter
            avg_geom_loss = geom_mse / loss_counter

        self.scheduler.step()
        return avg_prop_loss, avg_geom_loss
