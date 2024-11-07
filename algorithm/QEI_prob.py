import numpy as np
from typing import Any, Optional, Union

import torch
from torch import Tensor

import botorch
from botorch.models import FixedNoiseGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from botorch.fit import fit_gpytorch_mll
# Constrained Max Posterior Sampling s a new sampling class, similar to MaxPosteriorSampling,
# which implements the constrained version of Thompson Sampling described in [1].
from botorch.models.model_list_gp_regression import ModelListGP
from .base_optimizer import  GPOptimizer

from botorch.optim import optimize_acqf
from botorch.acquisition.objective import ConstrainedMCObjective

from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)

# Some constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NOISE_SE = 0.5
TRAIN_YVAR = torch.tensor(NOISE_SE**2, device=DEVICE, dtype=torch.float64)
NUM_RESTARTS = 10
MC_SAMPLES = 256

class QEI(GPOptimizer):
    def __init__(self, x, utils, safes, safe_threshold, bound, batch_size=1, 
                 eval=None, origin_to_latent=None, latent_to_origin=None, 
                 util_beta=2, safe_beta=2, dtype=torch.float64, device=DEVICE, eval_latent=True, latent_opt=True):
        
        # transform safe threshold to 0 and le inquality
        trans_safes = safes - safe_threshold
        trans_safes = -trans_safes
        
        super().__init__(x, utils, safes, safe_threshold, bound, batch_size, 
                         util_beta, safe_beta, dtype, device)
        
        self.eval = eval
        self.origin_to_latent = origin_to_latent
        self.latent_to_origin = latent_to_origin
        self.eval_latent = eval_latent
        self.latent_opt  = latent_opt
        
        self.device=device
        self.best_observed_ei = []
        # self.best_observed_nei = []
        self.best_random = []
        
        self.initial_state()
        # define a feasibility-weighted objective for optimization
        self.constrained_obj = ConstrainedMCObjective(
            objective=self.obj_callable,
            constraints=[self.constraint_callable],
        )
        self.bounds = torch.tensor([[0.0] * self.dim, [1.0] * self.dim], device=device, dtype=dtype)
        self.save_prefix = 'QEIPROB'
    
    def initialize_model(self, train_x, train_obj, train_con, state_dict=None):
        # define models for objective and constraint
        
        model_obj = FixedNoiseGP(train_x, train_obj, 
                                 TRAIN_YVAR.expand_as(train_obj)).to(train_x)
        model_con = FixedNoiseGP(train_x, train_con, 
                                 TRAIN_YVAR.expand_as(train_con)).to(train_x)
        # combine into a multi-output GP model
        model = ModelListGP(model_obj, model_con)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        
        # load state dict if it is passed
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll, model
    
    def generate_initial_data(self): ## need to change this n later
        # generate training data
        train_x = self.x
        exact_obj = self.utils  # add output dimension
        exact_con = self.safes  # add output dimension
        
        exact_obj = torch.tensor(exact_obj).to(device=self.device, dtype=self.dtype)
        exact_con = torch.tensor(exact_con).to(device=self.device, dtype=self.dtype)

        train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
        train_con = exact_con + NOISE_SE * torch.randn_like(exact_con)
        best_observed_value = self.weighted_obj(train_x).max().item()
        
        return train_x, train_obj, train_con, best_observed_value
    
    def initial_state(self):
        self.train_x_ei, self.train_obj_ei, self.train_con_ei, self.best_observed_value_ei = self.generate_initial_data()
        
        # Reshape so the values are 2d instead of only 1d
        self.train_x_ei = torch.tensor(self.train_x_ei).to(device=self.device, dtype=self.dtype)
        self.train_obj_ei = self.train_obj_ei.reshape(self.train_obj_ei.shape[0], 1)
        self.train_con_ei = self.train_con_ei.reshape(self.train_con_ei.shape[0], 1)
        
        self.mll_ei, self.model_ei = self.initialize_model(self.train_x_ei, self.train_obj_ei, self.train_con_ei)
        
        # self.train_x_nei, self.train_obj_nei, self.train_con_nei = self.train_x_ei, self.train_obj_ei, self.train_con_ei
        # best_observed_value_nei = self.best_observed_value_ei
        # self.mll_nei, self.model_nei = self.initialize_model(self.train_x_nei, self.train_obj_nei, self.train_con_nei)

        self.best_observed_ei.append(self.best_observed_value_ei)
        # self.best_observed_nei.append(best_observed_value_nei)
        self.best_random.append(self.best_observed_value_ei)
    
    def weighted_obj(self, cand_x):
        ret_val = cand_x
        for i in range(cand_x.shape[0]):
            if i < self.safe_threshold:
                ret_val[i] = 0
        
        return ret_val
    
    def optimize_acqf_and_get_observation(self, acq_func):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=self.batch_size,
            num_restarts=NUM_RESTARTS,
            raw_samples=self.x.shape[0],
            options={"batch_limit": 5, "maxiter": 200},
        )
        
        # observe new values
        new_x = candidates.detach()
        
        # problem with the way we are getting new obj and con
        if self.batch_size > 1:
            exact_obj, exact_con = [], []
            for i in range(self.batch_size):
                if not self.eval_latent:
                    if self.latent_opt:
                        eval_input = self.latent_to_origin(np.asarray(new_x[i].cpu().float()))
                    else:
                        eval_input = new_x[i].cpu().float()
                else:
                    if self.latent_opt:
                        eval_input = new_x[i].cpu().float()
                    else:
                        eval_input = self.origin_to_latent(np.asarray(new_x[i].cpu().float()))
                eval_input = eval_input.reshape(1, -1)

                exact_obj.append(self.eval(eval_input[0])[0])
                exact_con.append(self.eval(eval_input[0])[1])
        else:
            if not self.eval_latent:
                if self.latent_opt:
                    eval_input = self.latent_to_origin(np.asarray(new_x.cpu().float()))
                else:
                    eval_input = np.asarray(new_x.cpu().float())
            else:
                if self.latent_opt:
                    eval_input = np.asarray(new_x.cpu().float())
                else:
                    eval_input = self.origin_to_latent(np.asarray(new_x.cpu().float()))
            exact_obj, exact_con = self.eval(eval_input)
        
        exact_obj = torch.tensor(exact_obj).to(device=self.device, dtype=self.dtype)
        exact_con = torch.tensor(exact_con).to(device=self.device, dtype=self.dtype)
        
        new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
        new_con = exact_con + NOISE_SE * torch.randn_like(exact_con)
        
        return new_x, new_obj, new_con
    
    def obj_callable(self, Z):
        return Z[..., 0]

    def constraint_callable(self, Z):
        return Z[..., 1]
    
    # Needs to return x_next to match the output of the other algorithms
    def optimize(self):
        # Train the GP
        with botorch.settings.debug(True):
            try:
                fit_gpytorch_mll(self.mll_ei)
            except:
                pass
            # fit_gpytorch_mll(self.mll_nei)
    
        # define the qEI and qNEI acquisition modules using a QMC sampler
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        # for best_f, we use the best observed noisy values as an approximation
        qEI = qExpectedImprovement(
            model=self.model_ei,
            best_f=(self.train_obj_ei * (self.train_con_ei <= self.safe_threshold).to(self.train_obj_ei)).max(),
            const=sum(bool(x) for x in (self.train_con_ei <= self.safe_threshold))/len(self.train_con_ei),
            sampler=qmc_sampler,
            objective=self.constrained_obj,
        )
        
        # qNEI = qNoisyExpectedImprovement(
        #     model=self.model_nei,
        #     X_baseline=self.train_x_nei,
        #     sampler=qmc_sampler,
        #     objective=self.constrained_obj,
        # )
        
        self.new_x_ei, self.new_obj_ei, self.new_con_ei = self.optimize_acqf_and_get_observation(qEI.to(self.device, dtype=self.dtype))
        # self.new_x_nei, self.new_obj_nei, self.new_con_nei = self.optimize_acqf_and_get_observation(qNEI.to(self.device, dtype=self.dtype))
        
        
        return np.array(self.new_x_ei.cpu()) # ? should i return both?
    
    def update_state(self):
        # update training points
        self.train_x_ei = torch.cat([self.train_x_ei, self.new_x_ei])
        self.train_obj_ei = torch.cat([self.train_obj_ei, self.new_obj_ei.reshape(self.batch_size, 1)])
        self.train_con_ei = torch.cat([self.train_con_ei, self.new_con_ei.reshape(self.batch_size, 1)])

        # self.train_x_nei = torch.cat([self.train_x_nei, self.new_x_nei])
        # self.train_obj_nei = torch.cat([self.train_obj_nei, self.new_obj_nei.reshape(self.batch_size, 1)])
        # self.train_con_nei = torch.cat([self.train_con_nei, self.new_con_nei.reshape(self.batch_size, 1)])
        
        # update progress
        # best_random = update_random_observations(best_random)
        
        exact_obj_ei = self.train_obj_ei  # add output dimension
        # exact_obj_nei = self.train_obj_nei  # add output dimension
        
        best_value_ei = exact_obj_ei.max().item()
        # best_value_nei = exact_obj_nei.max().item()
        
        self.best_observed_ei.append(best_value_ei)
        # self.best_observed_nei.append(best_value_nei)

        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        self.mll_ei, self.model_ei = self.initialize_model(
            self.train_x_ei,
            self.train_obj_ei,
            self.train_con_ei,
            self.model_ei.state_dict(),
        )
        # self.mll_nei, self.model_nei = self.initialize_model(
        #     self.train_x_nei,
        #     self.train_obj_nei,
        #     self.train_con_nei,
        #     self.model_nei.state_dict(),
        # )
        
    def set_current_data(self, x, utils, safes):
        trans_safes = safes - self.safe_threshold
        trans_safes = -trans_safes
        super(QEI, self).set_current_data(x, utils, trans_safes)

class qExpectedImprovement(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        const:float = 1,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.const = const
        self.register_buffer("best_f", torch.as_tensor(best_f, dtype=float))

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.get_posterior_samples(posterior)
        obj = self.objective(samples, X=X)
        obj = (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
        q_ei = obj.max(dim=-1)[0].mean(dim=0)
        return self.const * q_ei