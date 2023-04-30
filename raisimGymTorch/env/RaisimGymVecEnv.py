# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import numpy as np


class RaisimGymVecEnv:

    def __init__(self, impl, cfg, normalize_ob=True, seed=0, normalize_rew=True, clip_obs=10.):
        self.normalize_ob = normalize_ob
        self.normalize_rew = normalize_rew
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.wrapper.init()
        self.num_obs = self.wrapper.getObDim()
        self.num_acts = self.wrapper.getActionDim()
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self.obs_rms = RunningMeanStd(shape=[self.num_envs, self.num_obs])
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=np.bool)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.displacements = np.zeros([self.num_envs, 4], dtype=np.float32)
        self.reward_info = np.zeros([self.num_envs, 16], dtype=np.float32)

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def set_command(self, command):
        self.wrapper.setCommand(command)

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action):
        self.wrapper.step(action, self._reward, self._done)
        return self._reward.copy(), self._done.copy()

    def load_scaling(self, dir_name, iteration, policy_type=None, num_g1=None):
        # policy_tupe 0 is the flat
        # policy type 1 is the combines
        # policy type 2 is the blind policy, from which we load encoders
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        loaded_mean = np.loadtxt(mean_file_name, dtype=np.float32)
        loaded_var = np.loadtxt(var_file_name, dtype=np.float32)
        #if policy_type == 0: (for cvpr)
        #    #self.obs_rms.mean[:,:loaded_mean.shape[1]] = loaded_mean
        #    #self.obs_rms.var[:,:loaded_var.shape[1]] = loaded_var
        #    self.obs_rms.mean[:,:loaded_mean.shape[1]] = loaded_mean
        #    self.obs_rms.var[:,:loaded_var.shape[1]] = loaded_var
        if policy_type == 0:
            assert num_g1 is not None, "You should provide num_g1"
            processed_mean = np.zeros_like(self.obs_rms.mean[:,self.obs_rms.mean.shape[1]//2:])
            processed_var = np.zeros_like(self.obs_rms.var[:,self.obs_rms.mean.shape[1]//2:])
            blind_mean = loaded_mean[:,loaded_mean.shape[1]//2:]
            blind_var = loaded_var[:,loaded_var.shape[1]//2:]
            g1_mean = blind_mean[:,-5:-3]
            g1_var = loaded_var[:,-5:-3]
            for i in range(num_g1+1):
                start = -3 - 2*i
                end = -1 - 2*i
                processed_mean[:,start:end] = g1_mean
                processed_var[:,start:end] = g1_var
            # copy action + prop+part
            processed_mean[:,:start] = blind_mean[:,:-5]
            processed_var[:,:start] = blind_var[:,:-5]
            # do not normalize slope
            processed_mean[:,-1] = 0.0
            processed_var[:,-1] = 1.0
            self.obs_rms.mean[:,:self.obs_rms.mean.shape[1]//2] = processed_mean
            self.obs_rms.var[:,:self.obs_rms.mean.shape[1]//2] = processed_var
        elif policy_type == 1:
            self.obs_rms.mean[:,self.obs_rms.mean.shape[1]//2:] = loaded_mean[:,self.obs_rms.mean.shape[1]//2:]
            self.obs_rms.var[:,self.obs_rms.mean.shape[1]//2:] = loaded_var[:,self.obs_rms.mean.shape[1]//2:]
        elif policy_type == 2:
            assert num_g1 is not None, "You should provide num_g1"
            processed_mean = np.zeros_like(self.obs_rms.mean[:,self.obs_rms.mean.shape[1]//2:])
            processed_var = np.zeros_like(self.obs_rms.var[:,self.obs_rms.mean.shape[1]//2:])
            blind_mean = loaded_mean[:,loaded_mean.shape[1]//2:]
            blind_var = loaded_var[:,loaded_var.shape[1]//2:]
            g1_mean = blind_mean[:,-5:-3]
            g1_var = loaded_var[:,-5:-3]
            for i in range(num_g1+1):
                start = -3 - 2*i
                end = -1 - 2*i
                processed_mean[:,start:end] = g1_mean
                processed_var[:,start:end] = g1_var
            # copy action + prop+part
            processed_mean[:,:start] = blind_mean[:,:-5]
            processed_var[:,:start] = blind_var[:,:-5]
            # do not normalize slope
            processed_mean[:,-1] = 0.0
            processed_var[:,-1] = 1.0
            self.obs_rms.mean[:,self.obs_rms.mean.shape[1]//2:] = processed_mean
            self.obs_rms.var[:,self.obs_rms.mean.shape[1]//2:] = processed_var
        else:
            # All other cases including dagger and so
            self.obs_rms.mean = loaded_mean
            self.obs_rms.var = loaded_var
        # duplicate
        #self.obs_rms.mean = np.tile(self.obs_rms.mean, [1,2])
        #self.obs_rms.var = np.tile(self.obs_rms.var, [1,2])


    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        np.savetxt(mean_file_name, self.obs_rms.mean)
        np.savetxt(var_file_name, self.obs_rms.var)

    def observe(self, update_mean=True):
        self.wrapper.observe(self._observation)

        if self.normalize_ob:
            if update_mean:
                self.obs_rms.update(self._observation)

            return self._normalize_observation(self._observation)
        else:
            return self._observation.copy()

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()

    def _normalize_observation(self, obs):
        if self.normalize_ob:

            return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -self.clip_obs,
                           self.clip_obs)
        else:
            return obs

    def get_dis(self):
        self.wrapper.getDis(self.displacements)
        return self.displacements.copy()
    
    def get_reward_info(self):
        self.wrapper.getRewardInfo(self.reward_info)
        return self.reward_info.copy()

    def reset_and_update_info(self):
        return self.reset(), self._update_epi_info()

    def _update_epi_info(self):
        info = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"r": eprew, "l": eplen}
            info[i]['episode'] = epinfo
            self.rewards[i].clear()

        return info

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def set_itr_number(self, itr_number):
        self.wrapper.setItrNumber(itr_number)

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def extra_info_names(self):
        return self._extraInfoNames


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr[:,self.mean.shape[1]//2:], axis=0)
        batch_var = np.var(arr[:,self.var.shape[1]//2:], axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        half_mean = self.mean[:,self.mean.shape[1]//2:]
        half_var = self.var[:,self.var.shape[1]//2:]
        delta = batch_mean - half_mean
        tot_count = self.count + batch_count

        new_mean = half_mean + delta * batch_count / tot_count
        m_a = half_var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * (self.count * batch_count / (self.count + batch_count))
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean[:,self.mean.shape[1]//2:] = new_mean
        self.var[:,self.var.shape[1]//2:] = new_var
        # account for the slope information
        self.mean[:,-1] = 0
        self.var[:,-1] = 1
        # duplicate future and present geometry
        self.mean[:,-5:-3] = self.mean[:,-3:-1]
        self.var[:,-5:-3] = self.var[:,-3:-1]
        self.count = new_count

