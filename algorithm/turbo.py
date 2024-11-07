import math
from dataclasses import dataclass
import torch
import time
import math
import json
import numpy as np
from copy import deepcopy
import torch
from torch.quasirandom import SobolEngine
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel, LinearKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import norm
from .base_optimizer import GPOptimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

class TuRBO(GPOptimizer):
    def __init__(
            self,
            x,
            utils,
            safes,
            safe_threshold,
            bound,
            batch_size=1,
            tr_num=1, # only support 1 trust region,
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
            util_beta,
            safe_beta,
            dtype,
            device
        )
        self.tr_num = tr_num
        self.initial_state()
        self.save_prefix = 'TuRBO'

    def initial_state(self):
        self.state = TurboState(
                    self.dim, batch_size=self.batch_size,
                    best_value=self.utils.max()
                    )
    




    def optimize(self, cand_size=5000):
        self.train_gp_model()
        # Scale the TR to be proportional to the lengthscales
        x = torch.from_numpy(self.x).to(dtype=self.dtype, device=self.device)
        x_center = x[self.utils.argmax(), :].clone()
        try:
            weights = self.util_gp.covar_module.base_kernel.lengthscale.squeeze().detach()
            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
            tr_lb = torch.clamp(x_center - weights * self.state.length / 2.0, 0.0, 1.0)
            tr_ub = torch.clamp(x_center + weights * self.state.length / 2.0, 0.0, 1.0)
        except:  # Linear kernel
            weights = 1
            tr_lb = torch.clamp(x_center - self.state.length / 2.0, 0.0, 1.0)
            tr_ub = torch.clamp(x_center + self.state.length / 2.0, 0.0, 1.0)


        sobol = SobolEngine(self.dim, scramble=True)
        pert = sobol.draw(cand_size).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dim, 1.0)
        # prob_perturb = 1
        mask = (
                torch.rand(cand_size, self.dim, dtype=dtype, device=device)
                <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, self.dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(cand_size, self.dim).clone()

        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        # thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        # with torch.no_grad():  # We don't need gradients when using TS
        #     X_next = thompson_sampling(X_cand, num_samples=batch_size)
        posterior = self.util_gp.posterior(X_cand)
        samples = posterior.rsample(sample_shape=torch.Size([self.batch_size]))
        # print('sample shape', samples.shape)
        samples = samples.reshape([self.batch_size, cand_size])
        Y_cand = samples.permute(1, 0)
        y_cand = Y_cand.detach().cpu().numpy()
        X_next = np.ones((self.batch_size, self.dim))
        for i in range(self.batch_size):
            # Pick the best point and make sure we never pick it again
            indbest = np.argmax(y_cand[:, i])
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            y_cand[indbest, :] = np.inf

        return X_next

    def update_state(self):
        update_state(self.state, self.utils[-self.batch_size:])
        if self.state.restart_triggered:
            self.restart = True








