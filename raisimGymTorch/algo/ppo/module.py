import torch.nn as nn
import numpy as np
import torch

class Expert:
    def __init__(self, policy, device='cpu', baseDim=46):
        self.policy = policy
        self.policy.to(device)
        self.device = device
        self.baseDim = baseDim 
        # 19 is size of priv info for the cvpr policy
        self.end_idx = baseDim + 19 #-self.geomDim*(self.n_futures+1) - 1

    def evaluate(self, obs):
        with torch.no_grad():
            resized_obs = obs[:, :self.end_idx]
            latent = self.policy.info_encoder(resized_obs[:,self.baseDim:])
            input_t = torch.cat((resized_obs[:,:self.baseDim], latent), dim=1)
            action = self.policy.action_mlp(input_t)
            return action

class Steps_Expert:
    def __init__(self, policy, device='cpu', baseDim=42, geomDim=2, n_futures=1, num_g1=8):
        self.policy = policy
        self.policy.to(device)
        self.device = device
        self.baseDim = baseDim 
        self.geom_dim = geomDim
        self.n_futures = n_futures
        # 23 is size of priv info for the cvpr policy
        self.privy_dim = 23
        self.end_prop_idx = baseDim + self.privy_dim
        self.g1_position = 0
        self.g1_slice = slice(self.end_prop_idx + self.g1_position * geomDim,
                              self.end_prop_idx + (self.g1_position+1) * geomDim)
        self.g2_slice = slice(-geomDim-1,-1)

    def evaluate(self, obs):
        with torch.no_grad():
            action = self.forward(obs)
            return action

    def forward(self, x):
        # get only the x related to the control policy
        x = x[:,:x.shape[1]//2]
        prop_latent = self.policy.prop_encoder(x[:,self.baseDim:self.end_prop_idx])
        geom_latents = []
        g1 = self.policy.geom_encoder(x[:,self.g1_slice])
        g2 = self.policy.geom_encoder(x[:,self.g2_slice])
        geom_latents = torch.hstack([g1,g2])
        return self.policy.action_mlp(torch.cat([x[:,:self.baseDim], prop_latent, geom_latents], 1))

class Actor:
    def __init__(self, architecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        try:
            self.architecture.to(device)
        except:
            print("If you're not in ARMA mode you have a problem")
        self.distribution.to(device)
        self.device = device

    def sample(self, obs):
        logits = self.architecture.architecture(obs)
        actions, log_prob = self.distribution.sample(logits)
        return actions.cpu().detach(), log_prob.cpu().detach()

    def evaluate(self, obs, actions):
        action_mean = self.architecture.architecture(obs)
        return self.distribution.evaluate(obs, action_mean, actions), action_mean

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs):
        return self.architecture.architecture(torch.from_numpy(obs).to(self.device))

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        transferred_graph = torch.jit.trace(self.architecture.architecture.to(device), example_input, check_trace=False)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape

class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs):
        return self.architecture.architecture(obs).detach()

    def evaluate(self, obs):
        return self.architecture.architecture(obs)

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Action_MLP_Encode(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, output_activation_fn = None, small_init= False, base_obdim = 45, geom_dim = 0,
                 n_futures = 3):
        super(Action_MLP_Encode, self).__init__()
        self.activation_fn = actionvation_fn
        self.output_activation_fn = output_activation_fn

        regular_obs_dim = base_obdim;
        self.regular_obs_dim = regular_obs_dim;
        self.geom_dim = geom_dim

        self.geom_latent_dim = 1
        self.prop_latent_dim = 8
        self.n_futures = n_futures
        
        # creating the action encoder
        modules = [nn.Linear(regular_obs_dim + self.prop_latent_dim + (self.n_futures + 1)*self.geom_latent_dim, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        action_output_layer = modules[-1]
        if self.output_activation_fn is not None:
            modules.append(self.output_activation_fn())
        self.action_mlp = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.action_mlp, scale)

        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

        #for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear)):
        #    module.weight.data *= 1e-6

    def forward(self, x):
        # get only the x related to the control policy
        geom_latent = x[:,-self.geom_dim:]
        prop_latent = x[:,-(self.geom_dim+self.prop_latent_dim):-self.geom_dim]
        return self.action_mlp(torch.cat([x[:,:self.regular_obs_dim], prop_latent, geom_latent], 1))

class Action_MLP_Encode_wrap(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, output_activation_fn = None,
                 small_init= False, base_obdim = 45, geom_dim = 0, n_futures = 3):
        super(Action_MLP_Encode_wrap, self).__init__()
        self.architecture = Action_MLP_Encode(shape, actionvation_fn, input_size, output_size, output_activation_fn, small_init, base_obdim, geom_dim, n_futures)
        self.input_shape = self.architecture.input_shape
        self.output_shape = self.architecture.output_shape

class MLPEncode(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, output_activation_fn = None, small_init= False, base_obdim = 45, geom_dim = 0,
                 n_futures = 3):
        super(MLPEncode, self).__init__()
        self.activation_fn = actionvation_fn
        self.output_activation_fn = output_activation_fn

        regular_obs_dim = base_obdim;
        self.regular_obs_dim = regular_obs_dim;
        self.geom_dim = geom_dim
        
        ## Encoder Architecture
        prop_latent_dim = 8
        geom_latent_dim = 1
        self.n_futures = n_futures
        self.prop_latent_dim = prop_latent_dim
        self.geom_latent_dim = geom_latent_dim
        self.prop_encoder =  nn.Sequential(*[
                                    nn.Linear(input_size - (n_futures+1)*geom_dim - regular_obs_dim -1, 256), self.activation_fn(),
                                    nn.Linear(256, 128), self.activation_fn(),
                                    nn.Linear(128, prop_latent_dim), self.activation_fn(),
                                    ]) 
        if self.geom_dim > 0:
            self.geom_encoder =  nn.Sequential(*[
                                        nn.Linear(geom_dim, 64), self.activation_fn(),
                                        nn.Linear(64, 16), self.activation_fn(),
                                        nn.Linear(16, geom_latent_dim), self.activation_fn(),
                                        ]) 
        else:
            raise IOError("Not implemented geom_dim")
        scale_encoder = [np.sqrt(2), np.sqrt(2), np.sqrt(2)]

        # creating the action encoder
        modules = [nn.Linear(regular_obs_dim + prop_latent_dim + (self.n_futures + 1)*geom_latent_dim, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        action_output_layer = modules[-1]
        if self.output_activation_fn is not None:
            modules.append(self.output_activation_fn())
        self.action_mlp = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.action_mlp, scale)
        self.init_weights(self.prop_encoder, scale_encoder)
        self.init_weights(self.geom_encoder, scale_encoder)
        if small_init: action_output_layer.weight.data *= 1e-6

        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

        #for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear)):
        #    module.weight.data *= 1e-6

    def forward(self, x):
        # get only the x related to the control policy
        if x.shape[1] > 130:
            # Hacky way to detect where you are (dagger or not)
            # TODO: improve on this!
            x = x[:,x.shape[1]//2:]
        prop_latent = self.prop_encoder(x[:,self.regular_obs_dim:-self.geom_dim*(self.n_futures+1) -1])
        geom_latents = []
        for i in reversed(range(self.n_futures+1)):
            start = -(i+1)*self.geom_dim -1
            end = -i*self.geom_dim -1
            if (end == 0): 
                end = None
            geom_latent = self.geom_encoder(x[:,start:end])
            geom_latents.append(geom_latent)
        geom_latents = torch.hstack(geom_latents)
        return self.action_mlp(torch.cat([x[:,:self.regular_obs_dim], prop_latent, geom_latents], 1))

class MLPEncode_wrap(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, output_activation_fn = None,
                 small_init= False, base_obdim = 45, geom_dim = 0, n_futures = 3):
        super(MLPEncode_wrap, self).__init__()
        self.architecture = MLPEncode(shape, actionvation_fn, input_size, output_size, output_activation_fn, small_init, base_obdim, geom_dim, n_futures)
        self.input_shape = self.architecture.input_shape
        self.output_shape = self.architecture.output_shape

class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size):
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps
        self.input_shape = input_size*tsteps
        self.output_shape = output_size
        # self.encoder = nn.Sequential(
        #         nn.Linear(input_size, 128), self.activation_fn(),
        #         nn.Linear(128, 32), self.activation_fn()
        #         )

        if tsteps == 50:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 8, stride = 4), nn.LeakyReLU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(), nn.Flatten())
            self.linear_output = nn.Sequential(
            nn.Linear(32 * 3, output_size), self.activation_fn()
            )
        elif tsteps == 10:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 2, stride = 1), nn.LeakyReLU(), 
                nn.Flatten())
            self.linear_output = nn.Sequential(
            nn.Linear(32 * 3, output_size), self.activation_fn()
            )
        elif tsteps == 20:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 6, stride = 2), nn.LeakyReLU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
                nn.Flatten())
            self.linear_output = nn.Sequential(
                nn.Linear(32 * 3, output_size), self.activation_fn()
            )
        else:
            raise NotImplementedError()



    def forward(self, obs):
        bs = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([bs * T, -1]))
        output = self.conv_layers(projection.reshape([bs, -1, T]))
        output = self.linear_output(output)
        return output

class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, output_activation_fn = None, small_init= False, base_obdim = None):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn
        self.output_activation_fn = output_activation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        action_output_layer = modules[-1]
        if self.output_activation_fn is not None:
            modules.append(self.output_activation_fn())
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        if small_init: action_output_layer.weight.data *= 1e-6

        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

        #for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear)):
        #    module.weight.data *= 1e-6

class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None

    def sample(self, logits):
        self.distribution = Normal(logits, self.std.reshape(self.dim))

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples).sum(dim=1)

        return samples, log_prob

    def evaluate(self, inputs, logits, outputs):
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std

class MultivariateGaussianDiagonalCovariance2(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance2, self).__init__()
        assert(dim == 12)
        self.dim = dim
        self.std_param = nn.Parameter(init_std * torch.ones(dim // 2))
        self.distribution = None

    def sample(self, logits):
        self.std = torch.cat([self.std_param[:3], self.std_param[:3], self.std_param[3:], self.std_param[3:]], dim=0)
        self.distribution = Normal(logits, self.std.reshape(self.dim))

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples).sum(dim=1)

        return samples, log_prob

    def evaluate(self, inputs, logits, outputs):
        self.std = torch.cat([self.std_param[:3], self.std_param[:3], self.std_param[3:], self.std_param[3:]], dim=0)
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std_param.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std_param.data = new_std
