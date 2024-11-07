import math
from dataclasses import dataclass
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from torch import Tensor
from .base_optimizer import GPOptimizer


@dataclass
class HdSafeBOState:
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
        # self.failure_tolerance = min(self.failure_tolerance, 5)
        # self.failure_tolerance = 4







class HdSafeBO(GPOptimizer):
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
            optimistic=True,
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
        self.initial_state()
        self.block_idx = 0
        self.global_samp = False
        self.optimistic = optimistic
        self.save_prefix = 'HdSafeBO' if self.optimistic else 'LSSO'
        self.save_prefix += f'_{util_beta}_{safe_beta}'

    def initial_state(self):
        self.state = HdSafeBOState(
            self.dim, batch_size=self.batch_size,
            best_value=-float('inf')
        )
    


    
    def get_safe_region(self, cand_x):
        safe_indices = np.ones(cand_x.shape[0], dtype=bool)
        for i, gp in enumerate(self.gps):
            gp.eval()
            observed_pred = gp.likelihood(gp(cand_x))
            lower, upper = observed_pred.confidence_region()
            means = observed_pred.mean.detach().cpu().numpy()
            var = (upper.detach().cpu().numpy()-lower.detach().cpu().numpy())/2
            if self.optimistic:
                ucb = means + self.safe_beta * var
                safe_indices[ucb < self.threshold[i]] = False
            else:
                print('no optimistic safe')

            
        return safe_indices

    def optimize(self, cand_size=5000):
        self.train_gp_model()
        x = self.x[self.block_idx:]
        utils = self.utils[self.block_idx:]
        safes = self.safes[self.block_idx:]
        safe_x = torch.from_numpy(x[safes>self.safe_threshold])\
            .to(dtype=self.dtype, device=self.device)
        safe_util = torch.from_numpy(utils[safes>self.safe_threshold])\
            .to(dtype=self.dtype, device=self.device)
        if len(safe_x)>0:
            x_center = safe_x[safe_util.argmax(), :]
            self.global_samp = False
        else: 
            # x_center = torch.rand(self.dim).to(dtype=self.dtype, device=self.device)
            x_center = x[utils.argmax(), :]
            x_center =  torch.from_numpy(x_center).to(dtype=self.dtype, device=self.device)

        if self.global_samp:
            print('global sampling')
        length = self.state.length
        
        # weights = self.util_gp.covar_module.base_kernel.lengthscale.squeeze().detach()
        # weights = weights / weights.mean()
        # weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        # tr_lb = torch.clamp(x_center - weights * length / 2.0, 0.0, 1.0)
        # tr_ub = torch.clamp(x_center + weights * length / 2.0, 0.0, 1.0)
        # print('weight', weights.max().item(), weights.min().item(), weights.mean().item(), weights.std().item())
        tr_lb = torch.clamp(x_center - length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + length / 2.0, 0.0, 1.0)
        
        # tr_lb = x_center - length / 2.0
        # tr_ub = x_center + length / 2.0
        # lb = torch.from_numpy(self.bound[0]).to(dtype=self.dtype, device=self.device)
        # ub = torch.from_numpy(self.bound[1]).to(dtype=self.dtype, device=self.device)
        while(True):
            sobol = SobolEngine(self.dim, scramble=True)
            pert = sobol.draw(cand_size).to(dtype=self.dtype, device=self.device)
            cand_x = tr_lb + (tr_ub - tr_lb) * pert
            safe_indices = self.get_safe_region(cand_x)

            if len(safe_indices[safe_indices>0]) < self.batch_size:
                length /= 2
                print(f'no safe, shrink {length}')
                tr_lb = torch.clamp(x_center - length / 2.0, 0.0, 1.0)
                tr_ub = torch.clamp(x_center + length / 2.0, 0.0, 1.0)
                if length <= self.state.length_min: # no safe point, select point from whole trust region
                    safe_indices = np.ones(cand_x.shape[0], dtype=bool)
                    self.global_samp = True
                    print('no safe, global samp')
                    break
            else:
                break


        posterior = self.util_gp.posterior(cand_x[safe_indices])
        # else:
        #     print('finding safe x...')
        #     posterior = self.safe_gp.posterior(cand_x[safe_indices])

        samples = posterior.rsample(sample_shape=torch.Size([self.batch_size]))
        del posterior
        samples = samples.reshape([self.batch_size, cand_x[safe_indices].shape[0]])
        Y_cand = samples.permute(1,0)
        del samples
        y_cand = Y_cand.detach().cpu().numpy()
        del Y_cand
        X_next = torch.zeros(min(self.batch_size, len(safe_indices)), self.dim).to(device=self.device, dtype=self.dtype)
        max_indices = []
        for k in range(self.batch_size):
            j = np.argmax(y_cand[:, k])
            # print('select', i, j)
            X_next[k] = cand_x[safe_indices][j]
            max_indices.append(j)
            assert np.isfinite(y_cand[j, k])  # Just to make sure we never select nan or inf
            # Make sure we never pick this point again
            y_cand[j, :] = -np.inf
        
        return X_next.detach().cpu().numpy()
        # return X_next.detach().cpu().numpy(), safe_indices, [max_idx.detach().cpu().numpy()]

    def update_state(self):

        # if not self.global_samp:
        Y_next = torch.from_numpy(self.utils[-self.batch_size:]).to(device=self.device, dtype=self.dtype)
        C_next = torch.from_numpy(self.safes[-self.batch_size:]).to(device=self.device, dtype=self.dtype)
        bool_tensor = C_next >= self.safe_threshold
        bool_tensor = torch.all(bool_tensor, dim=-1)
        Valid_Y_next = Y_next[bool_tensor]
        Valid_C_next = C_next[bool_tensor]
        if len(Valid_Y_next)<self.batch_size: # not all safe
            self.state.success_counter = 0
            self.state.failure_counter += 1
        if len(Valid_Y_next)>0:
            if Valid_Y_next.max() > self.state.best_value + 1e-3 * math.fabs(self.state.best_value):
                self.state.best_value = max(self.state.best_value, Valid_Y_next.max().item())
                if len(Valid_Y_next) == self.batch_size:
                    self.state.success_counter += 1
                    self.state.failure_counter = 0

        if self.state.success_counter == self.state.success_tolerance:  # Expand trust region
            self.state.length = min(2.0 * self.state.length, self.state.length_max)
            self.state.success_counter = 0
        elif self.state.failure_counter == self.state.failure_tolerance:  # Shrink trust region
            self.state.length /= 2.0
            self.state.failure_counter = 0

        if self.state.length < self.state.length_min:  # Restart when trust region becomes too small
            # self.state.restart_triggered = True
            self.state.length = 0.8
            self.global_samp = True
        # if self.global_samp:
            # self.block_idx = len(self.x)-self.batch_size
            print(f'global sampling {self.block_idx}')
        print(f'length {self.state.length} {self.state.failure_counter} {self.state.failure_tolerance}')



    
