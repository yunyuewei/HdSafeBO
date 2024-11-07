import os
import sys
import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.append('./')
from .base_task import BaseTask
from .gpfun.gp_fun import GPFunction


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ae_model')))




class GPTask(BaseTask):
    def __init__(
            self,
            algorithm,
            dim=100,
            latent_dim=20,
            fun_dim = 10,
            noise = 0.01,
            init_data_path=None,
            model_path=None,  # List, VAE model and score model
            safe_threshold=-0.75,
            eval_latent=False,
            latent_opt=True,
            batch_size=1,
            dtype=torch.float64,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.init_num = 200
        self.latent_opt = latent_opt
        self.fun_dim = fun_dim
        self.util_fun = GPFunction(ls=0.05, dim=self.fun_dim)
        self.safe_fun = GPFunction(ls=0.05, dim=self.fun_dim)
        
        self.direction = np.random.normal(size=(dim, latent_dim))
        self.direction /= np.linalg.norm(self.direction)
        
        self.eff_dim = np.random.choice(range(latent_dim), self.fun_dim)
        self.noise = noise

        super().__init__(
            algorithm,
            dim,
            latent_dim,
            init_data_path,
            model_path,
            safe_threshold,
            eval_latent,
            latent_opt,
            batch_size,
            dtype,
            device
        )


    def load_model(self, model_path):
        self.pca = PCA(n_components=self.latent_dim)

    def set_bound(self):
        # Set bounds
        self.latent_lb = np.zeros(self.latent_dim)
        self.latent_ub = np.ones(self.latent_dim)
        self.latent_bound = (self.latent_lb, self.latent_ub)

        self.lb = self.x.min() * np.ones(self.dim)
        self.ub = self.x.max() * np.ones(self.dim)
        self.bound = (self.lb, self.ub)

    def load_init_data(self, init_data_path):
        '''
        Load initial data if has data path, or sample inside the bound.
        Transfrom initial data to the latent space if optimize in the latent space.
        '''
        if init_data_path is not None:
            # Does not have an init data path
            raise NotImplementedError
     

        self.latent_x =  np.random.random((self.init_num, self.latent_dim))
        self.x = self.latent_x@ np.linalg.pinv(self.direction)
        self.pca.fit(self.x)
        util, safe = self.eval_batch(self.x)  # for test
        self.utils = util  # ndarray (n, )
        self.safes = safe
        self.latent_x = self.origin_to_latent(self.x)


        print(f'init {self.safes.min()} {self.utils[self.safes > self.safe_threshold].max()}')

    def origin_to_latent(self, x):
        return self.pca.transform(x)

    def latent_to_origin(self, x):
        return self.pca.inverse_transform(x)

    def eval(self, x):
        '''
        Evaluate function value of given x.
        '''
        # print(x.shape, self.direction.shape)
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        
        lx = x.reshape(1, -1) @ self.direction
        lx = torch.from_numpy(lx).to(dtype=self.dtype, device=self.device)
        
        ex = lx[:, self.eff_dim].reshape(1, -1)
        util = self.util_fun(ex).item()
        safe = self.safe_fun(ex).item()

        return util, safe