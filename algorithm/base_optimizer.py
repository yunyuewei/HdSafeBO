import os
import numpy as np
import torch
import gpytorch
from botorch.models import SingleTaskGP
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model


class BaseOptimizer:
    '''
    Optimizer on given input and bounds 
    '''

    def __init__(
            self,
            x,
            utils,
            safes,
            safe_threshold,
            bound,
            batch_size=1,
            dtype=torch.float64,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.x = x
        self.dim = x.shape[1]
        self.utils = utils
        self.safes = safes
        self.safe_threshold = safe_threshold
        self.bound = bound
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device
        self.batch_size = batch_size
        self.restart = False
        self.save_prefix = f'basealgorithm'

    def optimize(self):
        '''
        Optimize based on history data
        '''
        pass

    def set_current_data(self, x, utils, safes):
        self.x = x
        self.utils = utils
        self.safes = safes

    def update_state(self):
        pass


def fit_gp(train_x, train_y, dtype, device, norm=True, fit=True):
    X_torch = torch.from_numpy(train_x).to(dtype=dtype, device=device)
    Y_torch = torch.from_numpy(train_y).to(dtype=dtype, device=device).unsqueeze(-1)

    if norm:
        Y_torch = (Y_torch - Y_torch.mean()) / Y_torch.std()
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    dim = X_torch.shape[-1]
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
    )
    gp = SingleTaskGP(X_torch, Y_torch, covar_module=covar_module, likelihood=likelihood)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    if fit:
        with gpytorch.settings.max_cholesky_size(-float('inf')):
            # Fit the model
            fit_gpytorch_model(mll)
    # print(gp.covar_module.base_kernel.lengthscale)
    return gp


class GPOptimizer(BaseOptimizer):
    '''
    Optimizer based on Gaussian Process
    '''

    def __init__(
            self,
            x,
            utils,
            safes,
            safe_threshold,
            bound,
            batch_size=1,
            util_beta=2,
            safe_beta=2,
            dtype=torch.float64,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super().__init__(
            x,
            utils,
            safes,
            safe_threshold,
            bound,
            batch_size,
            dtype,
            device
        )

        # self.train_gp_model()
        self.util_beta = util_beta
        self.safe_beta = safe_beta

    def train_gp_model(self, fit=True, max_len=10000):
        # Transform values and threshold to list
        self.vals = [self.utils]
        self.threshold = [-float('inf')]
        if isinstance(self.safes, list):
            assert isinstance(self.safe_threshold, list)
            self.vals.extend(self.safes)
            self.threshold.extend(self.safe_threshold)
        else:
            self.vals.append(self.safes)
            self.threshold.append(self.safe_threshold)

        self.util_gp = fit_gp(self.x, self.utils, self.dtype, self.device, fit=fit)
        self.gps = [self.util_gp]
        if isinstance(self.safes, list):
            self.safe_gp = []
            for safe in self.safes:
                self.safe_gps.append(fit_gp(self.x, safe, self.dtype, self.device, fit=fit))
            self.gps.extend(self.safe_gp)
        else:
            self.safe_gp = fit_gp(self.x, self.safes, self.dtype, self.device, norm=False)
            self.gps.append(self.safe_gp)






