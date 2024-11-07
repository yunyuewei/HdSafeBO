import math
from dataclasses import dataclass
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from torch import Tensor
from .base_optimizer import GPOptimizer








class CONFIG(GPOptimizer):
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
        self.block_idx = 0
        self.global_samp = False
        self.optimistic = optimistic
        self.save_prefix = 'CONFIG'
        self.save_prefix += f'_{util_beta}_{safe_beta}'


    
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
            else:
                ucb = means - self.safe_beta * var

            safe_indices[ucb < self.threshold[i]] = False

            
        return safe_indices

    def optimize(self, cand_size=5000):
        self.train_gp_model()
        x = self.x[self.block_idx:]
        utils = self.utils[self.block_idx:]
        safes = self.safes[self.block_idx:]
        
        sobol = SobolEngine(self.dim, scramble=True)

        cand_x = sobol.draw(cand_size).to(dtype=self.dtype, device=self.device)
        safe_indices = self.get_safe_region(cand_x)

        print(f'unsafe len {len(safe_indices[safe_indices==0])}')

        posterior = self.util_gp.posterior(cand_x[safe_indices])

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
    


    
