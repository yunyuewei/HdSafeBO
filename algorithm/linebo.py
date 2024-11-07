import math
from dataclasses import dataclass
import numpy as np
import torch
import gpytorch
from torch.quasirandom import SobolEngine
from .safeopt import SafeOpt
from .base_optimizer import fit_gp
class LineBO(SafeOpt):
    def __init__(
            self,
            x,
            utils,
            safes,
            direction,
            safe_threshold,
            bound,
            batch_size=1,
            util_beta=2,
            safe_beta=2,
            optimistic=False,
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
            optimistic, 
            dtype,
            device
        )
        
        
        self.block_idx = 0
        self.global_samp = False
        self.optimistic = optimistic
        self.save_prefix = 'LineBO'
        self.save_prefix += f'_{util_beta}_{safe_beta}'
        self.get_unique_input()
    
    def get_safe_region(self, cand_x):
        safe_indices = np.ones(cand_x.shape[0], dtype=bool)
        for i, gp in enumerate(self.gps):
            gp.eval()

            observed_pred = gp.likelihood(gp(cand_x))
            lower, upper = observed_pred.confidence_region()
            means = observed_pred.mean.detach().cpu().numpy()
            var = (upper.detach().cpu().numpy() - lower.detach().cpu().numpy()) / 2
            lcb = means - self.safe_beta * var

            safe_indices[lcb < self.threshold[i]] = False


        return safe_indices
    
    def get_unique_input(self):
        # if 'Line' in self.save_prefix:
        X_torch = torch.from_numpy(self.x).to(dtype=self.dtype, device=self.device)
        # print(X_torch[:, 0])
        # print('before', len(X_torch))
        # remove same input
        same_index = torch.zeros(len(X_torch))
        dist = torch.cdist(X_torch, X_torch) + torch.eye(len(X_torch)).to(dtype=self.dtype, device=self.device)
        # print('min', dist.min())
        for i in range(len(X_torch)):
            if torch.any(dist[i, i+1:] == 0):
                same_index[i] = 1
        same_index = same_index.detach().cpu().numpy()
        self.train_x = self.x[same_index == 0]
        self.train_utils = self.utils[same_index == 0]
        print('remove', len(same_index[same_index == 1]))
        if isinstance(self.safes, list):
            self.train_safes = []
            for i, safe in enumerate(self.safes):
                self.train_safes.append(self.safes[i][same_index == 0])
        else:
            self.train_safes = self.safes[same_index == 0]
        # else:
        #     self.train_x = self.x
        #     self.train_utils = self.utils
        #     self.train_safes = self.safes
        
    def set_current_data(self, x, utils, safes):
        super().set_current_data(x, utils, safes)
        self.get_unique_input() # avoid same train input

    def train_gp_model(self, fit=True):
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

        self.util_gp = fit_gp(self.train_x, self.train_utils, self.dtype, self.device, fit=fit)
        self.gps = [self.util_gp]
        if isinstance(self.safes, list):
            self.safe_gp = []
            for i, safe in enumerate(self.safes):
                self.safe_gps.append(fit_gp(self.train_x, self.train_safes[i], self.dtype, self.device, fit=fit))
            self.gps.extend(self.safe_gp)
        else:
            self.safe_gp = fit_gp(self.train_x, self.train_safes, self.dtype, self.device, norm=False)
            self.gps.append(self.safe_gp)


    def optimize(self, cand_size=5000):
        
        self.train_gp_model()
        dim = self.x.shape[1]
        sobol = SobolEngine(self.x.shape[1], scramble=True)
 
        cand_x = sobol.draw(cand_size).to(dtype=self.dtype, device=self.device)
        
        # safe_indices = self.get_feasible_region(cand_x)
        safe_indices = self.get_safe_region(cand_x)

        print(f'safe len {len(safe_indices[safe_indices > 0])}')
        if len(safe_indices[safe_indices > 0]) < self.batch_size:
            
            
            if len(safe_indices[safe_indices > 0]) > 0:
                print("safe indices smaller than batch_size")
                X_next = cand_x[safe_indices > 0].detach().cpu().numpy()
            else:
                X_next = np.zeros((0, dim))
                print('No safe points in the candidate set, choosing current best safe point')
            safe_x = self.x[self.safes>self.safe_threshold]
            safe_utils = self.utils[self.safes>self.safe_threshold]
            if len(safe_x) < self.batch_size-len(X_next):
                X_next = np.vstack((X_next, safe_x))
                best_x = safe_x[safe_utils.argmax()]
                while len(X_next) < self.batch_size:
                    X_next = np.vstack((X_next, best_x))
            else:
                while len(X_next) < self.batch_size:
                    best_x = safe_x[safe_utils.argmax()]
                    X_next = np.vstack((X_next, best_x))
                    safe_utils[safe_utils.argmax()] = -float('inf')
            # print('next', X_next)
            return X_next
        
        posterior = self.util_gp.posterior(cand_x[safe_indices])

        samples = posterior.rsample(sample_shape=torch.Size([self.batch_size]))
        del posterior
        samples = samples.reshape([self.batch_size, cand_x[safe_indices].shape[0]])
        Y_cand = samples.permute(1, 0)
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
        # print('next', X_next)
        return X_next.detach().cpu().numpy()