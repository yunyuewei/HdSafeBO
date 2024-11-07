import os
import sys
from enum import Enum
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import gpytorch
from torchvision import datasets
import torchvision.transforms as transforms

sys.path.append('./')
from .digit.cnn_model import VAE, ScoreNet
from .base_task import BaseTask
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ae_model')))

from models import load_pretrained

class DigitTask(BaseTask):
    def __init__(
            self,
            algorithm,
            dim=784,
            task_idx=[0, 1, 2], # List, valid numbers
            latent_dim=20,
            init_data_path=None,
            model_path=None, # List, VAE model and score model
            bound_path=None,
            safe_threshold=0.5,
            eval_latent=False,
            latent_opt=True,
            batch_size=1,
            dtype=torch.float64,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.task_idx = task_idx
        self.init_num = 20
        self.bound_path = bound_path
        self.latent_opt = latent_opt
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
        assert isinstance(model_path, list)
        vae_path, score_path = model_path
        if vae_path is not None and score_path is not None:
            # self.model = VAE(x_dim=784, h_dim1=512,
            #                       h_dim2=256, z_dim=self.latent_dim).to(dtype=self.dtype, device=self.device)
            # vae_state_dict = torch.load(vae_path, map_location=self.device)
            # self.model.load_state_dict(vae_state_dict)
            self.model, _ = load_pretrained(
                identifier=vae_path[0],
                config_file=vae_path[1],
                ckpt_file='model_best.pkl',
                root='./ae_model'
            )
            self.model = self.model.to(dtype=self.dtype, device=self.device)
            self.score_model = ScoreNet().to(dtype=self.dtype, device=self.device)
            cnn_state_dict = torch.load(score_path, map_location=self.device)
            self.score_model.load_state_dict(cnn_state_dict)
        else:
            raise NotImplementedError

    def set_bound(self):
        # Set bounds
        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)
        self.bound = (self.lb, self.ub)

        #  Set latent bounds according to initial data , for now
        # TODO: look at previous paper to see how to set bounds
        bound_data = np.load(self.bound_path)
        self.latent_lb = bound_data['latent_lb']
        self.latent_ub = bound_data['latent_ub']
        self.latent_bound = (self.latent_lb, self.latent_ub)
        self.latent_bound = (self.latent_lb, self.latent_ub)

    def load_init_data(self, init_data_path):
        '''
        Load initial data if has data path, or sample inside the bound.
        Transfrom initial data to the latent space if optimize in the latent space.
        '''
        if init_data_path is None:
            raise NotImplementedError

        test_dataset = datasets.MNIST(root=init_data_path, train=False, transform=transforms.ToTensor(),
                                      download=True)


        target_idx = torch.zeros(0).to(dtype=int, device=self.device)
        for idx in self.task_idx:
            t_idx = torch.argwhere(test_dataset.targets==idx).to(dtype=int, device=self.device)
            target_idx = torch.cat((target_idx, t_idx), 0)
        # rand_idx = np.random.choice(len(target_idx), self.init_num)
        target_idx = target_idx[:self.init_num]
        self.x = test_dataset.data[target_idx.cpu()].reshape(self.init_num, self.dim)/255 # ndarray (n, dim)
        

        # self.x = test_dataset.data.reshape(10000, self.dim) # ndarray (n, dim)

        # for idx in range(10):
        #     t_idx = torch.argwhere(test_dataset.targets==idx)
        #     x = test_dataset.data[t_idx]
        #     util, safe = self.eval_batch(x)
        #     plt.figure()
        #     plt.subplot(121)
        #     plt.hist(util)
        #     plt.subplot(122)
        #     plt.hist(safe)
        #     plt.title(str(idx))
        #     plt.show()

        # raise NotImplementedError
        # self.x = torch.zeros(0, self.dim)
        # pert = torch.rand(init_num, self.latent_dim)
        # lb = torch.ones(self.latent_dim) * -4
        # ub = torch.ones(self.latent_dim) * 4
        # pert = lb + (ub - lb) * pert
        #
        # recon_x = self.latent_to_origin(pert)
        self.x = self.x.detach().cpu().numpy()


        util, safe = self.eval_batch(self.x)  # for test
        self.utils = util  # ndarray (n, )
        self.safes = safe

        self.latent_x = self.origin_to_latent(self.x)

        print(f'init {self.safes.min()} {self.utils[self.safes>self.safe_threshold].max()}')

    def origin_to_latent(self, x):
        x = torch.from_numpy(x).reshape(-1, 1, 28, 28).to(dtype=self.dtype, device=self.device)
        return self.model.encode(x).detach().cpu().numpy()
    
    def latent_to_origin(self, x):
        z = torch.from_numpy(x).to(dtype=self.dtype, device=self.device)
        return self.model.decode(z).reshape(len(z), self.dim).detach().cpu().numpy()


    def score_image(self, x, temp=200):
        """The input x is an image and an expected score
        based on the CNN classifier and the scoring
        function is returned.
        """
        with torch.no_grad():
            probs = torch.nn.functional.softmax(self.score_model(x)/temp, dim=1)  # b x 10
        return probs

    def eval(self, x):
        '''
        Evaluate function value of given x.
        '''

        if isinstance(x, torch.Tensor):
            img = x.to(dtype=self.dtype, device=self.device).reshape(1, 1, 28, 28)
        else: # ndarray
            img = torch.from_numpy(x).to(dtype=self.dtype, device=self.device).reshape(1, 1, 28, 28)

        score = self.score_image(img*255).flatten().detach().cpu().numpy()
        valid_score = score[self.task_idx]
        util = (img).sum().item()
        safe = valid_score.max()
        return util, safe