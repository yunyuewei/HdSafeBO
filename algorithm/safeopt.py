import math
from dataclasses import dataclass
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from torch import Tensor
from .base_optimizer import GPOptimizer


class SafeOpt(GPOptimizer):
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
            dtype,
            device
        )
        self.block_idx = 0
        self.global_samp = False
        self.optimistic = optimistic
        self.save_prefix = 'OLSSO' if self.optimistic else 'PLSSO'
        self.save_prefix += f'_{util_beta}_{safe_beta}'

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

    def get_feasible_region(self, cand_x):
        safe_indices = self.get_safe_region(cand_x)

        w = torch.zeros(2, len(cand_x)).to(dtype=self.dtype, device=self.device)

        lcb = np.zeros((len(self.gps), len(cand_x)))
        ucb = np.zeros((len(self.gps), len(cand_x)))
        for i, gp in enumerate(self.gps):
            # if i == 0:
            #     continue
            gp.eval()
            observed_pred = gp.likelihood(gp(cand_x))
            means = observed_pred.mean.detach().cpu().numpy()
            lower, upper = observed_pred.confidence_region()
            var = (upper - lower)
            w[i] = var
            if i == 0:
                lcb[i] = means - self.util_beta * var.detach().cpu().numpy()
                ucb[i] = means + self.util_beta * var.detach().cpu().numpy()
            else:
                lcb[i] = means - self.safe_beta * var.detach().cpu().numpy()
                ucb[i] = means + self.safe_beta * var.detach().cpu().numpy()

        max_w = torch.max(w, dim=0).values
        # Only work for 1 constraint now
        safe_lcb = lcb[1].copy()
        safe_lcb[safe_indices == 0] = -float('inf')
        util_lcb = lcb[0].copy()
        util_lcb[safe_indices == 0] = -float('inf')
        max_lcb = util_lcb.max()
        max_indices = np.zeros(cand_x.shape[0], dtype=bool)  # Maximizer
        exp_indices = np.zeros(cand_x.shape[0], dtype=bool)  # Expander

        max_indices[ucb[0] > max_lcb] = 1
        # print('max', len(max_indices[max_indices==1]))
        max_indices[safe_indices == 0] = 0

        old_x = self.x.copy()
        old_utils = self.utils.copy()
        old_safes = self.safes.copy()
        try:
            max_mw = max_w[max_indices == 1].max()
        except:
            max_mw = -float('inf')
        for i, x in enumerate(cand_x):
            if safe_indices[i] == 0 or max_indices[i] == 1 or max_w[i] < max_mw:
                # print (safe_indices[i], max_indices[i], max_w[i], max_mw)
                continue
            new_x = np.vstack((old_x, x.detach().cpu().numpy().reshape(1, -1)))
            new_utils = np.hstack((old_utils,
                                   torch.from_numpy(ucb[0, i].reshape(-1)).to(dtype=self.dtype, device=self.device).cpu()
                                   ))
            new_safes = np.hstack((old_safes,
                                   torch.from_numpy(ucb[1, i].reshape(-1)).to(dtype=self.dtype, device=self.device).cpu()
                                   ))
            self.set_current_data(new_x, new_utils, new_safes)
            self.train_gp_model(fit=False)  # do not update parameter
            new_safe_indices = self.get_safe_region(cand_x)
            new_safe_indices[safe_indices == 1] = 1
            # print(f'new {len(safe_indices[safe_indices==1])}, {len(new_safe_indices[new_safe_indices==1])} ')
            if new_safe_indices.sum() > safe_indices.sum():
                exp_indices[i] = 1
            self.set_current_data(old_x, old_utils, old_safes)
        self.train_gp_model()
        cand_indices = np.zeros(cand_x.shape[0], dtype=bool)

        cand_indices[max_indices == 1] = 1
        cand_indices[exp_indices == 1] = 1
        if exp_indices.sum() == 0 and max_indices.sum() == 0:
            cand_indices[safe_indices == 1] = 1

        return cand_indices

    def optimize(self, cand_size=5000):
        self.train_gp_model()
        # x = self.x[self.block_idx:]
        # utils = self.utils[self.block_idx:]
        # safes = self.safes[self.block_idx:]
        # safe_x = torch.from_numpy(x[safes>self.safe_threshold])\
        #     .to(dtype=self.dtype, device=self.device)
        # safe_util = torch.from_numpy(utils[safes>self.safe_threshold])\
        #     .to(dtype=self.dtype, device=self.device)

        sobol = SobolEngine(self.dim, scramble=True)

        cand_x = sobol.draw(cand_size).to(dtype=self.dtype, device=self.device)
        safe_indices = self.get_feasible_region(cand_x)

        print(f'safe len {len(safe_indices[safe_indices > 0])}')
        if len(safe_indices[safe_indices > 0]) == 0:
            print('No safe points in the candidate set, choosing current best safe point')
            
            safe_x = self.x[self.safes>self.safe_threshold]
            safe_utils = self.utils[self.safes>self.safe_threshold]
            if len(safe_x) < self.batch_size:
                X_next = safe_x
                best_x = safe_x[safe_utils.argmax()]
                while len(X_next) < self.batch_size:
                    X_next = np.vstack((X_next, best_x))
            else:
                X_next = np.zeros((0, self.dim))
                while len(X_next) < self.batch_size:
                    best_x = safe_x[safe_utils.argmax()]
                    X_next = np.vstack((X_next, best_x))
                    safe_utils[safe_utils.argmax()] = -float('inf')
            print(self.x.shape)
            print("Xnext shape", X_next.shape)
            return X_next
            # raise NotImplementedError

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

        return X_next.detach().cpu().numpy()
        # return X_next.detach().cpu().numpy(), safe_indices, [max_idx.detach().cpu().numpy()]



