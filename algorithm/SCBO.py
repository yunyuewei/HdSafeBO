import math
import os
import warnings
from dataclasses import dataclass

import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine

from botorch.fit import fit_gpytorch_mll
# Constrained Max Posterior Sampling s a new sampling class, similar to MaxPosteriorSampling,
# which implements the constrained version of Thompson Sampling described in [1].
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from .base_optimizer import  GPOptimizer

@dataclass
class ScboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    best_constraint_values: Tensor = torch.ones(2,) * torch.inf
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_tr_length(state):
    # Update the length of the trust region according to
    # success and failure counters
    # (Just as in original TuRBO paper)
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:  # Restart when trust region becomes too small
        state.restart_triggered = True

    return state


def update_state(state, Y_next, C_next):
    """Method used to update the TuRBO state after each step of optimization.

    Success and failure counters are updated according to the objective values
    (Y_next) and constraint values (C_next) of the batch of candidate points
    evaluated on the optimization step.

    As in the original TuRBO paper, a success is counted whenver any one of the
    new candidate points improves upon the incumbent best point. The key difference
    for SCBO is that we only compare points by their objective values when both points
    are valid (meet all constraints). If exactly one of the two points being compared
    violates a constraint, the other valid point is automatically considered to be better.
    If both points violate some constraints, we compare them inated by their constraint values.
    The better point in this case is the one with minimum total constraint violation
    (the minimum sum of constraint values)"""

    # Determine which candidates meet the constraints (are valid)
    C_next = C_next.reshape(len(C_next), 1)
    bool_tensor = C_next <= 0
    bool_tensor = torch.all(bool_tensor, dim=-1)
    Valid_Y_next = Y_next[bool_tensor]
    Valid_C_next = C_next[bool_tensor]
    print(f'valid {bool_tensor}')
    # raise NotImplementedError
    if Valid_Y_next.numel() == 0:  # if none of the candidates are valid
        # pick the point with minimum violation
        sum_violation = C_next.sum(dim=-1)
        min_violation = sum_violation.min()
        # if the minimum voilation candidate is smaller than the violation of the incumbent
        if min_violation < state.best_constraint_values.sum():
            # count a success and update the current best point and constraint values
            state.success_counter += 1
            state.failure_counter = 0
            # new best is min violator
            state.best_value = Y_next[sum_violation.argmin()].item()
            state.best_constraint_values = C_next[sum_violation.argmin()]
        else:
            # otherwise, count a failure
            state.success_counter = 0
            state.failure_counter += 1
    else:  # if at least one valid candidate was suggested,
        # throw out all invalid candidates
        # (a valid candidate is always better than an invalid one)

        # Case 1: if the best valid candidate found has a higher objective value that
        # incumbent best count a success, the obj valuse has been improved
        improved_obj = max(Valid_Y_next) > state.best_value + 1e-3 * math.fabs(
            state.best_value
        )
        # Case 2: if incumbent best violates constraints
        # count a success, we now have suggested a point which is valid and thus better
        obtained_validity = torch.all(state.best_constraint_values > 0)
        improved_obj = torch.any(improved_obj>0)
        try:
            if improved_obj or obtained_validity:  # If Case 1 or Case 2
                # count a success and update the best value and constraint values
                state.success_counter += 1
                state.failure_counter = 0
                state.best_value = max(Valid_Y_next).item()
                state.best_constraint_values = Valid_C_next[Valid_Y_next.argmax()]
            else:
                # otherwise, count a failure
                state.success_counter = 0
                state.failure_counter += 1
        except:
            print(improved_obj)
            print(obtained_validity)
            raise NotImplementedError

    # Finally, update the length of the trust region according to the
    # updated success and failure counters
    state = update_tr_length(state)
    return state

class SCBO(GPOptimizer):
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
        # transform safe threshold to 0 and le inquality
        trans_safes = safes - safe_threshold
        trans_safes = -trans_safes


        super().__init__(
            x,
            utils,
            trans_safes,
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
        self.save_prefix = 'SCBO'

    def initial_state(self):
        
        self.state = ScboState(
                    self.dim, batch_size=self.batch_size,
                    )

    def set_current_data(self, x, utils, safes):
        trans_safes = safes - self.safe_threshold
        trans_safes = -trans_safes
        super(SCBO, self).set_current_data(x, utils, trans_safes)

# # Define example state
# state = ScboState(dim=dim, batch_size=batch_size)
# print(state)

    def optimize(self, cand_size=5000):
        self.train_gp_model()
        X = torch.from_numpy(self.x).to(dtype=self.dtype, device=self.device)
        Y = torch.from_numpy(self.utils).to(dtype=self.dtype, device=self.device)
        # assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

        # Create the TR bounds

        x_center = X[Y.argmax(), :].clone() # Maybe some problem in original implementation?
        tr_lb = torch.clamp(x_center - self.state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + self.state.length / 2.0, 0.0, 1.0)

        # Thompson Sampling w/ Constraints (SCBO)
        dim = X.shape[-1]
        sobol = SobolEngine(dimension=self.dim, scramble=True)
        pert = sobol.draw(cand_size).to(dtype=self.dtype, device=self.device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(cand_size, dim, dtype=self.dtype, device=self.device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=self.device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(cand_size, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points using Constrained Max Posterior Sampling
        constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
            model=self.util_gp, constraint_model=ModelListGP(self.safe_gp), replacement=False
        )
        with torch.no_grad():
            X_next = constrained_thompson_sampling(X_cand, num_samples=self.batch_size)

        return X_next.detach().cpu().numpy()

    def update_state(self):
        y_next = torch.from_numpy(self.utils[-self.batch_size:]).to(dtype=self.dtype, device=self.device)
        c_next = torch.from_numpy(self.safes[-self.batch_size:]).to(dtype=self.dtype, device=self.device)

        update_state(self.state, y_next, c_next)
        print(f'state {self.state.length}')
        if self.state.restart_triggered:
            self.restart = True