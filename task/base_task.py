import time
import torch
import numpy as np
from algorithm.hdsafebo import HdSafeBO
from algorithm.turbo import TuRBO
from algorithm.SCBO import SCBO
from algorithm.QEI import QEI
from algorithm.QEI_prob import QEI as QEI_PROB
from algorithm.CMAES import CMAES 
from algorithm.CONFIG import CONFIG
from algorithm.safeopt import SafeOpt
from algorithm.linebo import LineBO
from task.utils import from_unit_cube, to_unit_cube

class BaseTask:
    def __init__(
            self, 
            algorithm, 
            dim, 
            latent_dim=None, 
            init_data_path=None, 
            model_path=None,
            safe_threshold = -float('inf'),
            eval_latent=False,
            latent_opt=True,
            batch_size=1,
            dtype=torch.float64,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ):
        
        self.dim = dim
        self.latent_dim = latent_dim
        self.dtype=dtype
        self.device=device
        self.safe_threshold = safe_threshold
        self.algorithm = algorithm
        self.batch_size = batch_size
        self.restart_idx = 0
        self.eval_latent = eval_latent
        self.latent_opt = latent_opt
        self.linebo = True if algorithm == 'LineBO' else False
        if self.linebo:
            ori_dim = self.latent_dim if latent_opt else self.dim
            self.line_direction = np.random.normal(ori_dim, 1)

        self.load_model(model_path)
        self.load_init_data(init_data_path)
        self.set_bound()
        self.set_opt_space()
        self.init_algo()
        self.optimize_time = np.zeros(0)
        
        
        

    def set_bound(self):
        '''
        Set search bound of original and latent space
        '''
        self.bound = None # ndarray (dim, 2)
        self.latent_bound = None # ndarray (latent_dim, 2)
        

    def load_init_data(self, init_data_path):
        '''
        Load initial data if has data path, or sample inside the bound.
        Transfrom initial data to the latent space if optimize in the latent space.
        '''
        self.x = None # ndarray (n, dim)
        self.utils = None # ndarray (n, )
        self.safes = None
        self.latent_x = None  # ndarray (n, latent_dim)
        pass

    def sample_init_data(self):
        pert = torch.rand(self.init_num, self.latent_dim).detach().cpu().numpy()
        lb = self.latent_bound[0]
        ub = self.latent_bound[1]
        pert = lb + (ub - lb) * pert

        latent_x = pert

        x = self.latent_to_origin(latent_x)

        util, safe = self.eval_batch(x)

        self.x = np.vstack((self.x, x))
        self.latent_x = np.vstack((self.latent_x, latent_x))
        self.utils = np.hstack((self.utils, util))
        self.safes = np.hstack((self.safes, safe))


    def load_model(self, model_path):
        '''
        Load model if has model path, or optimize in the original space.
        '''
        self.model = None
        # self.latent_opt = False
        pass

    def set_opt_space(self):
        if self.linebo:
            if self.latent_opt:
                self.opt_x = to_unit_cube(self.latent_x, self.latent_bound[0], self.latent_bound[1])
                self.ori_dim = self.latent_dim
                self.safe_x = self.latent_x[self.safes>self.safe_threshold]
            else:
                self.opt_x = to_unit_cube(self.x, self.bound[0], self.bound[1])
                self.ori_dim = self.dim
                self.safe_x = self.opt_x[self.safes>self.safe_threshold]
            self.best_x = self.safe_x[self.utils[self.safes>self.safe_threshold].argmax()].reshape(1, -1)
            # print('best', self.best_x)
            
            
            # Coordnate alignment implementation
            self.opt_dir = np.random.randint(self.latent_dim if self.latent_opt else self.dim)    
            
            print('dir', self.opt_dir)
            # After getting the direction, project x to this direction
            # self.opt_x = self.opt_x[:, self.opt_dir].reshape(-1, 1)
            # if self.latent_opt:
            self.line_direction = np.random.normal(size=(self.ori_dim, 1))
            # self.line_direction /= 
            # self.line_direction *= self.d[1]latent_boun[0]-self.latent_bound[0][0]
            
            self.opt_x = np.dot(self.opt_x-self.best_x, self.line_direction)/np.linalg.norm(self.line_direction)**2
            
            self.opt_bound = (np.ones(1) * self.opt_x.min(), np.ones(1) * self.opt_x.max())
            self.opt_x =  to_unit_cube(self.opt_x, self.opt_bound[0], self.opt_bound[1])
            # else:
            #     self.opt_bound = (self.bound[0][self.opt_dir], self.bound[1][self.opt_dir])
            self.opt_dim = 1
            
        elif self.latent_opt:
            # algorithm optimize over an unit cube
            self.opt_x = to_unit_cube(self.latent_x, self.latent_bound[0], self.latent_bound[1])
            self.opt_bound = (np.zeros(self.latent_dim), np.ones(self.latent_dim))
            self.opt_dim = self.latent_dim
        else:
            self.opt_x = to_unit_cube(self.x, self.bound[0], self.bound[1])
            self.opt_bound = (np.zeros(self.latent_dim), np.ones(self.latent_dim))
            self.opt_dim = self.dim

    def init_algo(self):
        '''
        Initial algorithm state.
        '''
        if self.algorithm == 'HdSafeBO':
            self.algo = HdSafeBO(self.opt_x[self.restart_idx:],
                              self.utils[self.restart_idx:],
                              self.safes[self.restart_idx:],
                              self.safe_threshold, self.opt_bound, self.batch_size,
                                                              safe_beta=2,
                            )
        elif self.algorithm == 'LSSO':
            self.algo = HdSafeBO(self.opt_x[self.restart_idx:],
                              self.utils[self.restart_idx:],
                              self.safes[self.restart_idx:],
                              self.safe_threshold, self.opt_bound, self.batch_size,optimistic=False,
                                                              safe_beta=2,
                            )
        elif self.algorithm == 'PLSSO':
            self.algo = HdSafeBO(self.opt_x[self.restart_idx:],
                              self.utils[self.restart_idx:],
                              self.safes[self.restart_idx:],
                              self.safe_threshold, self.opt_bound, self.batch_size,
                              optimistic=False)
        elif self.algorithm == 'TuRBO':
            self.algo = TuRBO(self.opt_x[self.restart_idx:],
                              self.utils[self.restart_idx:],
                              self.safes[self.restart_idx:],
                              self.safe_threshold, self.opt_bound, self.batch_size)
        elif self.algorithm == 'SCBO':
            self.algo = SCBO(self.opt_x[self.restart_idx:],
                              self.utils[self.restart_idx:],
                              self.safes[self.restart_idx:],
                              self.safe_threshold, self.opt_bound, self.batch_size)
        elif self.algorithm == 'QEI':
            self.algo = QEI(self.opt_x[self.restart_idx:],
                            self.utils[self.restart_idx:],
                            self.safes[self.restart_idx:],
                            self.safe_threshold, self.opt_bound, self.batch_size, 
                            self.eval, self.origin_to_latent, self.latent_to_origin, 
                            eval_latent=self.eval_latent, latent_opt=self.latent_opt)
            # we need the eval and origin to latent for qei
        elif self.algorithm == 'QEI_PROB':
            self.algo = QEI_PROB(self.opt_x[self.restart_idx:],
                                 self.utils[self.restart_idx:],
                                 self.safes[self.restart_idx:],
                                 self.safe_threshold, self.opt_bound, self.batch_size, 
                                 self.eval, self.origin_to_latent, self.latent_to_origin, 
                                 eval_latent=self.eval_latent, latent_opt=self.latent_opt)
        elif self.algorithm == 'CMAES':
            self.algo = CMAES(self.opt_x[self.restart_idx:],
                              self.utils[self.restart_idx:],
                              self.safes[self.restart_idx:],
                              self.safe_threshold, self.opt_bound, self.batch_size, 
                              self.eval, self.origin_to_latent, self.latent_to_origin, 
                              eval_latent=self.eval_latent, latent_opt=self.latent_opt)
        elif self.algorithm == 'CONFIG':
            self.algo = CONFIG(self.opt_x[self.restart_idx:],
                              self.utils[self.restart_idx:],
                              self.safes[self.restart_idx:],
                              self.safe_threshold, self.opt_bound, self.batch_size,
                                                              safe_beta=2,
                            )
        elif self.algorithm == 'SafeOpt':
            self.algo = SafeOpt(self.opt_x[self.restart_idx:],
                              self.utils[self.restart_idx:],
                              self.safes[self.restart_idx:],
                              self.safe_threshold, self.opt_bound, self.batch_size,
                                                              safe_beta=2,
                            )
        elif self.algorithm == 'LineBO':
            self.linebo = True
            self.algo = LineBO(self.opt_x[self.restart_idx:],
                              self.utils[self.restart_idx:],
                              self.safes[self.restart_idx:],
                              self.line_direction, self.safe_threshold, self.opt_bound, 
                              self.batch_size, safe_beta=2,
                            )
        else:
            self.algo = None
        

    def origin_to_latent(self, x):
        pass
    
    def latent_to_origin(self, x):
        pass

    def eval(self, x):
        '''
        Evaluate function value of given x. Specify the computation here.
        '''
        pass

    def eval_batch(self, X):
        '''
        Evaluate function value of given batch of x.
        '''
        eval_vals = np.array([self.eval(x) for x in X])
        return eval_vals[:, 0], eval_vals[:, 1]

    def algo_optimize(self):
        if self.algo is not None:
            x_next = self.algo.optimize()
            # x_next_ori = np.copy(x_next)
            # print(x_next)
            if self.linebo:
                # if self.latent_opt:
                #     x_next = from_unit_cube(x_next, np.ones(1) * self.latent_bound[0][self.opt_dir], np.ones(1) * self.latent_bound[1][self.opt_dir])
                # else:
                #     x_next = from_unit_cube(x_next, np.ones(1) * self.bound[0][self.opt_dir], np.ones(1) * self.bound[1][self.opt_dir])
                x_next = from_unit_cube(x_next, self.opt_bound[0], self.opt_bound[1])
                new_x_next = np.broadcast_to(self.best_x, (self.batch_size, self.ori_dim)).copy()
                # print(x_next, self.opt_bound)
                
                # new_x_next[:, self.opt_dir] = x_next.ravel()
                new_x_next = self.best_x + x_next * self.opt_dir
                
                # raise NotImplementedError
                x_next = new_x_next
                # print(x_next)
                # raise NotImplementedError
            if self.latent_opt:
                x_next = from_unit_cube(x_next, self.latent_bound[0], self.latent_bound[1])
            else:
                x_next = from_unit_cube(x_next, self.bound[0], self.bound[1])
            # print(x_next)
            # raise NotImplementedError
        else: # for test
            x_next = np.random.random((self.batch_size, self.opt_dim))
            lb, ub = self.opt_bound
            x_next = lb + (ub - lb) * x_next
        if not self.eval_latent:
            if self.latent_opt:
                eval_x = self.latent_to_origin(x_next)
            else:
                eval_x = x_next
        else:
            if self.latent_opt:
                eval_x = x_next
            else:
                eval_x = self.origin_to_latent(x_next)

        # print(eval_x[5], self.latent_lb.shape, self.latent_ub.shape)
        # data = np.load('results/2024-06-03-12-54-47_HdSafeBO_2_2_MuscleTask_5_0_act_data4_True/0.npz')
       
        # policy = data['latent_x'][205]
        # print(eval_x[5]-policy)
        # raise NotImplementedError
        util_next, safe_next = self.eval_batch(eval_x)
        print(f'values this round: {util_next} {safe_next}')
        return x_next, util_next, safe_next

    def state_update(self):
        '''
        Update algorithm or model state.
        '''
        if self.algo is not None:
            self.algo.set_current_data(
                self.opt_x[self.restart_idx:],
                self.utils[self.restart_idx:],
                self.safes[self.restart_idx:])
            self.algo.update_state()
            if self.algo.restart:
                print('restart')
                self.restart_idx = len(self.utils)
                self.sample_init_data()
                self.set_opt_space()
                self.init_algo()



        else: # for test
            pass

    def optimize(self):
        time_start = time.time()
        x_next, util_next, safe_next = self.algo_optimize()
        self.optimize_time = np.append(self.optimize_time, time.time()-time_start)
        
        if self.latent_opt:
            origin_x = self.latent_to_origin(x_next)
            latent_x = x_next
        else:
            origin_x = x_next
            latent_x = self.origin_to_latent(x_next)
        if isinstance(self.x, list):
            self.x.extend(origin_x)
        else:
            self.x = np.vstack((self.x, origin_x))
        self.latent_x = np.vstack((self.latent_x, latent_x))
        self.utils = np.hstack((self.utils, util_next))
        self.safes = np.hstack((self.safes, safe_next))
        self.set_opt_space()
        self.state_update()
    
    def save_result(self, folder, rep):
        save_path = f'./results/{folder}/{rep}'
        # np.savez(f'{save_path}.npz', x=self.x, latent_x=self.latent_x, 
        #          utils=self.utils, safes=self.safes, threshold=self.safe_threshold, init_num=self.init_num)
        np.savez(f'{save_path}.npz', 
                #  latent_x=self.latent_x, 
                 utils=self.utils, safes=self.safes, threshold=self.safe_threshold, init_num=self.init_num, opt_time=self.optimize_time)


        

        