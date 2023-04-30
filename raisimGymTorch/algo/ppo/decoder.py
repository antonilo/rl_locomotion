from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from .storage import ObsStorage 

class DecoderTrainer:
    def __init__(self,
            decoder,
            num_envs, 
            num_transitions_per_env,
            latent_size,
            decoder_output_size,
            num_learning_epochs=4,
            num_mini_batches=4,
            device=None,
            learning_rate=5e-4):

        self.decoder = decoder
        self.storage = ObsStorage(num_envs, num_transitions_per_env, [latent_size], [decoder_output_size], device)
        self.optimizer = optim.Adam([*self.decoder.parameters()], lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.device = device

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.loss_fn = nn.MSELoss()

    # def observe(self, obs):
    #     with torch.no_grad():
    #         actions = self.actor.get_student_action(torch.from_numpy(obs).to(self.device))
    #     return actions.detach().cpu().numpy()

    def step(self, expected_params, student_latent):
        self.storage.add_obs(student_latent.detach().cpu().numpy(), expected_params.detach())

    def update(self):
        # Learning step
        mse_loss = self._train_step()
        self.storage.clear()
        return mse_loss

    def _train_step(self):
        for epoch in range(self.num_learning_epochs):
            # return loss in the last epoch
            mse = 0
            loss_counter = 0
            for latent, expected_params in self.storage.mini_batch_generator_inorder(self.num_mini_batches):

                predicted = self.decoder.architecture(latent)
                loss = self.loss_fn(predicted, expected_params)

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                mse += loss.item()
                loss_counter += 1

            avg_loss = mse / loss_counter

        self.scheduler.step()
        return avg_loss


    def save_deterministic_graph(self, fname_encoder, example_input, device='cpu'):
        encoder_graph = torch.jit.trace(self.decoder.architecture.to(device), example_input)
        torch.jit.save(encoder_graph, fname_encoder)
        self.decoder.to(self.device)
        