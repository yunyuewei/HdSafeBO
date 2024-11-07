# import myosuite
import gym
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings

os.environ['GYM_ENV'] = 'gym'
os.environ['MUJOCO_GL'] = 'egl'
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
plt.rcParams["font.family"] = "Latin Modern Roman"



class SynergyWrapper(gym.ActionWrapper):
    """
    gym.ActionWrapper that reformulates the action space as the synergy space and inverse transforms
    synergy-exploiting actions back into the original muscle activation space.
    """

    def __init__(self, env, pca):
        super().__init__(env)
        self.pca = pca
        self.action_space = gym.spaces.Box(low=-3., high=3., shape=(self.pca.components_.shape[0],), dtype=np.float32)
        

    def action(self, act):
        action = self.pca.inverse_transform([act])
        action = np.clip(action, 0, 1)
        return action[0]
    

class SynergyAEWrapper(gym.ActionWrapper):
    """
    gym.ActionWrapper that reformulates the action space as the synergy space and inverse transforms
    synergy-exploiting actions back into the original muscle activation space.
    """

    def __init__(self, env, model, dtype, device, syn_num=5):
        super().__init__(env)
        self.model = model
        self.dtype = dtype
        self.device = device

        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(syn_num,), dtype=np.float32)
        

    def action(self, act):
        # action = self.pca.inverse_transform(self.ica.inverse_transform(self.scaler.inverse_transform([act])))
        act_tensor = torch.from_numpy(act).reshape(1, -1).to(dtype=self.dtype, device=self.device)
        action = self.model.decode(act_tensor).detach().cpu().numpy()
        action = np.clip(action, 0, 1)
        # print('mus', action)
        # raise NotImplementedError
        return action[0]

