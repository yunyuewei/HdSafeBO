import numpy as np
import torch
import matplotlib.pyplot as plt
from task.muscle.muscle_utils import *
# from task.muscle.task_env import MuscleSynergyEnv
from task.muscle.task_env_mjhand import MuscleSynergyEnv
from .base_task import BaseTask



class MuscleTask(BaseTask):
    def __init__(
            self,
            algorithm,
            dim=None, #String with variable length
            latent_dim=None,
            init_data_path=None,
            model_path=None, # List, [ica,  pca, nomalizer]
            bound_path=None,
            safe_threshold=-4.4,
            syn_num = 5,
            env_type='hand',
            eval_latent=True,
            latent_opt=True,
            batch_size=1,
            mode='Linear',
            dtype=torch.float64,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        self.init_num = 200
        self.syn_num = syn_num
        self.mode = mode
        self.bound_path = bound_path
        eval_latent = latent_opt
        self.env_type = env_type
        print('latent_opt', latent_opt, eval_latent)

        super().__init__(
            algorithm,
            dim,
            latent_dim,
            init_data_path,
            model_path,
            safe_threshold,
            eval_latent,
            latent_opt,
            batch_size,
            dtype,
            device
        )

    def load_model(self, model_path):
        '''
        Load model if has model path, or optimize in the original space.
        '''
        if model_path is not None:
            self.model = MuscleSynergyEnv(self.syn_num, model_path, env_type=self.env_type, mode=self.mode,latent_opt=self.latent_opt ,dtype=self.dtype, device=self.device)
            self.dim = self.model.origin_dim
            self.latent_dim = self.model.dim
            
        else:
            raise NotImplementedError

    def set_bound(self):
        # Set bounds TODO: set original bound

        self.lb = 0 * np.ones(self.dim)
        self.ub = 1 * np.ones(self.dim)
        self.bound = (self.lb, self.ub)
        if self.mode == 'Linear':
            if self.env_type == 'hand':
                self.latent_lb = -3 * np.ones(self.latent_dim)
                self.latent_ub = 3 * np.ones(self.latent_dim)  
            else:
                if isinstance(self.bound_path, list):
                    bound_data = np.load(self.bound_path[0])
                else:
                    bound_data = np.load(self.bound_path)
                    
                lb_act = bound_data['lb']
                ub_act = bound_data['ub']
                print(lb_act, ub_act)

                lb = np.zeros(0)
                ub = np.zeros(0)
                for i in range(self.model.act_shape):
                    lb = np.hstack((lb, lb_act[i] * np.ones(self.model.obs_shape)))
                    ub = np.hstack((ub, ub_act[i] * np.ones(self.model.obs_shape)))
                self.latent_lb = lb
                self.latent_ub = ub
        else: 
            bound_data = np.load(self.bound_path)
            if self.latent_opt:
                lb_act = bound_data['lb']
                ub_act = bound_data['ub']
                print(lb_act, ub_act)

                lb = np.zeros(0)
                ub = np.zeros(0)
                for i in range(self.model.act_shape):
                    lb = np.hstack((lb, lb_act[i] * np.ones(self.model.obs_shape)))
                    ub = np.hstack((ub, ub_act[i] * np.ones(self.model.obs_shape)))
                self.latent_lb = lb
                self.latent_ub = ub
            else:
                self.latent_lb = -20 * np.ones(self.latent_dim)
                self.latent_ub = 20 * np.ones(self.latent_dim)  
        
        self.latent_bound = (self.latent_lb, self.latent_ub)
        if self.latent_opt:
            self.model.set_bound(self.latent_lb, self.latent_ub)
            # set shrinked bound using top data
            if isinstance(self.bound_path, list) and self.env_type == 'leg':
                bound_data = np.load(self.bound_path[1])
                self.latent_lb = bound_data['latent_lb']
                self.latent_ub = bound_data['latent_ub']
                print('set new opt bound', self.latent_lb[:self.syn_num], self.latent_ub[:self.syn_num])
        else:
            self.model.set_bound(self.lb, self.ub)

    def load_init_data(self, init_data_path):
        '''
        Load initial data if has data path, or sample inside the bound.
        Transfrom initial data to the latent space if optimize in the latent space.
        '''
        if init_data_path is None:
            raise NotImplementedError
        else:
            init_data = np.load(init_data_path)

            self.latent_x = init_data['x'][:self.init_num]
            self.x = self.latent_to_origin(self.latent_x)
            # util, safe = self.eval_batch(self.x)  # for test
            self.utils = init_data['r'][:self.init_num]  # ndarray (n, )
            self.safes = init_data['v'][:self.init_num]

            print("train util shape:", self.utils.shape)
            print("train safe shape:", self.safes.shape)

            print(f"train x list length: {len(self.x)}\n")
            print(len(self.safes[self.safes>-4.5]))
            # raise NotImplementedError
            if np.all(self.safes<self.safe_threshold):
                print(f'init {self.dim} {self.latent_dim} {self.safes.min()} no safe yet')
            else:
                print(f'init {self.dim} {self.latent_dim} {self.safes.min()} {self.utils[self.safes>self.safe_threshold].max()} {np.max(self.latent_x)}')

    def eval(self, x):
        '''
        Evaluate function value of given x.
        '''
        # syn = self.model.mus_to_syn([x])[0]
        # print(x)
        # raise NotImplementedError
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.array([x])
        # print(x.shape)
        # latent_x = self.origin_to_latent(x)
        util, safe = self.model(x)

        return util, safe
    
    def origin_to_latent(self, x):
        return self.model.mus_to_syn_policy(x)
    
    def latent_to_origin(self, x):
        return self.model.syn_to_mus_policy(x)
