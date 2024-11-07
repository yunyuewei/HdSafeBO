import random
from typing import Any
import torch
import gpytorch
from gpytorch.constraints.constraints import Interval
import matplotlib.pyplot as plt
import pickle as pkl


class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dim=None, kernel='Matern'):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        lengthscale_constraint = Interval(0.005, 10.0)  # [0.005, sqrt(dim)]

        if train_x is None:
            assert dim is not None
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=dim,
                                              lengthscale_constraint=lengthscale_constraint)
            )
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[1],
                                              lengthscale_constraint=lengthscale_constraint)
            )


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def sample_function(self, x_cand):
        self.eval()
        y_cand = self.likelihood(self(x_cand)).sample(torch.Size([1]))
        return y_cand.ravel()


class GPFunction:
    def __init__(
            self,
            dim,
            ls,
            kernel='Matern',
            dtype=torch.float64,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        # self.ls = 0.05 + random.random()*(0.1-0.05)
        # self.ls = 0.1
        self.ls = ls
        self.dim = dim
        self.dtype = dtype
        self.device = device
        self.kernel = kernel
        self.get_new_gp(None, None, self.ls)
        self.eval_x = torch.zeros(0, dim).to(dtype=self.dtype, device=self.device)
        self.eval_y = torch.zeros(0).to(dtype=self.dtype, device=self.device)

    def get_new_gp(self, train_x, train_y, ls):
        noise_constraint = Interval(0, 0.2)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)
        self.gp = GP(train_x=train_x, train_y=train_y, likelihood=likelihood, dim=self.dim, kernel=self.kernel)
        hypers = {}
        hypers["covar_module.outputscale"] = 1.0
        hypers["covar_module.base_kernel.lengthscale"] = ls
        hypers["likelihood.noise"] = 1e-4
        self.gp.initialize(**hypers)
        self.gp.to(dtype=self.dtype, device=self.device)

    def __call__(self, x):
        x = x.to(dtype=self.dtype, device=self.device)
        repeat = False

        if len(self.eval_x) > 0 and len(x) == 1:
            nearest_idx = torch.argmin(torch.norm(self.eval_x - x, dim=1))
            nearest_val = torch.min(torch.norm(self.eval_x - x, dim=1))
            if nearest_val == 0:
                repeat = True
            # print(nearest_idx, nearest_val)
            # raise NotImplementedError
        if not repeat:
            y = self.gp.sample_function(x)
            # print(self.eval_y.shape, y.shape)

            self.eval_x = torch.cat((self.eval_x, x), 0)
            self.eval_y = torch.cat((self.eval_y, y), 0)
            self.get_new_gp(self.eval_x, self.eval_y, self.ls)
        else:
            # print('repeat')
            y = self.eval_y[nearest_idx:nearest_idx + 1]
        return y.detach().cpu()