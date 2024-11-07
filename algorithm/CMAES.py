import numpy as np
import torch
from cma import CMAEvolutionStrategy

from .base_optimizer import  GPOptimizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CMAES(GPOptimizer):
    def __init__(self, x, utils, safes, safe_threshold, bound, batch_size=1, 
                eval=None, origin_to_latent=None, latent_to_origin=None, 
                util_beta=2, safe_beta=2, dtype=torch.float64, device=DEVICE, 
                eval_latent=True, latent_opt=True):
    
        # transform safe threshold to 0 and le inquality
        trans_safes = safes - safe_threshold
        trans_safes = -trans_safes
        
        super().__init__(x, utils, safes, safe_threshold, bound, batch_size, 
                            util_beta, safe_beta, dtype, device)
        
        self.eval             = eval
        self.origin_to_latent = origin_to_latent
        self.latent_to_origin = latent_to_origin
        self.eval_latent      = eval_latent
        self.latent_opt       = latent_opt
        self.bounds           = torch.tensor([[0.0] * self.dim, [1.0] * self.dim], device=device, dtype=dtype)
        self.save_prefix      = 'CMAES'

        # Do initialization
        self.restart=False
        self.initialize()
        
    def initialize(self):
        self.domain = ContinuousDomain(np.array(self.bounds.cpu()[0]), np.array(self.bounds.cpu()[1]))
        
        self.x0 = self.domain.l + self.domain.range/2
        self.sigma0 = 0.2
        
        if self.batch_size > 1:
            self.cma = CMAEvolutionStrategy(x0=self.x0, sigma0=self.sigma0,  inopts={'bounds': [0,1], 'popsize': self.batch_size})
        else:
            self.cma = CMAEvolutionStrategy(x0=self.x0, sigma0=self.sigma0,  inopts={'bounds': [0,1]})
        self.initial_data = self.x
        self._exit = False
        
        self._X = None
        self._X_i = 0
        self._Y = None
        self.t  = 0
        
        self.best_x = None
        self.best_y = None
    
    def _optimize(self):
        if self._X is None:
            # get new population
            if self.batch_size > 1:
                self._X = self.cma.ask(number=self.batch_size)
            else:
                self._X = self.cma.ask()
            self._Y = np.empty(len(self._X))
            self._X_i = 0

        return self._X
    
    def optimize(self):
        
        next_x = self._optimize()
        
        if isinstance(next_x, tuple):
            x = next_x[0]
            additional_data = next_x[1]
        else:
            x = next_x
            additional_data = {}
        additional_data['t'] = self.t
        self.t += 1

        # for continous domains, check if x is inside box
        if isinstance(self.domain, ContinuousDomain):
            if (x > self.domain.u).any() or (x < self.domain.l).any():
                # logger.warning(f'Point outside domain. Projecting back into box.\nx is {x}, with limits {self.domain.l}, {self.domain.u}')
                x = np.maximum(np.minimum(x, self.domain.u), self.domain.l)

        
        self.x = np.asarray([l.tolist() for l in x])
        
        return self.x
    
    def update_state(self):
        y = []
        # print("_X", len(self._X))
        x = self.x[len(self.x)-len(self._X):len(self.x)]
        
        for i in range(len(x)):
            if not self.eval_latent:
                if self.latent_opt:
                    eval_input = self.latent_to_origin(x[i].reshape(1, x[i].shape[0]))
                else:
                    eval_input = x[i].reshape(1, x[i].shape[0])
            else:
                if self.latent_opt:
                    eval_input = x[i].reshape(1, x[i].shape[0])
                else:
                    eval_input = self.origin_to_latent(x[i].reshape(1, x[i].shape[0]))
            
            y.append(self.eval(eval_input[0])[0])
        
        # self._Y[self._X_i] = y
        self._X_i += 1

        # population complete
        if self._X_i == len(self._X):
            self.cma.tell(self._X, [-i for i in y])
            self._X = None
    
    def set_current_data(self, x, utils, safes):
        trans_safes = safes - self.safe_threshold
        trans_safes = -trans_safes
        super(CMAES, self).set_current_data(x, utils, trans_safes)
        
            
# Helper class:
class ContinuousDomain:

    def __init__(self, l, u, denormalized_domain=None):
        # TODO make sure everything is a numpy array
        self._l = l
        self._u = u
        self._range = self._u - self._l
        self._d = l.shape[0]
        self._bounds = np.vstack((self._l,self._u)).T

    @property
    def l(self):
        return self._l

    @property
    def u(self):
        return self._u

    @property
    def bounds(self):
        return self._bounds

    @property
    def range(self):
        return self._range

    @property
    def d(self):
        return self._d

    def normalize(self, x):
        return (x - self._l)/self._range

    def denormalize(self, x):
        return x * self._range + self._l

    def project(self, X):
        """
        Project X into domain rectangle.
        """
        return np.minimum(np.maximum(X, self.l), self.u)

    @property
    def is_continuous(self):
        return True