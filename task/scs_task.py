import os
import sys
from enum import Enum
import json
import numpy as np
import pandas as pd
import torch
import gpytorch
sys.path.append('./')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ae_model')))
from models import load_pretrained

from .base_task import BaseTask

def pole_data_transfer(origin_pole_dis, ):
    opt_index = np.arange(1, 17)
    pos_top = np.array([[6,3],[12,3],[18,3],[24,3],[30,3],[36,3],[42,3],[48,3],
                    [3,6],[9,6],[15,6],[21,6],[27,6],[33,6],[39,6],[45,6],
                    [3,9],[9,9],[15,9],[21,9],[27,9],[33,9],[39,9],[45,9],
                    [6,12],[12,12],[18,12],[24,12],[30,12],[36,12],[42,12],[48,12]])
    pos_top = pos_top-1
    
    x_new = np.zeros((1,52,14))
    y = np.zeros((1,52,14))
    pole_dis_new = np.zeros((1, 33))
    null_count = 0
    
    pole_dis = origin_pole_dis[opt_index-1]
    # print('pole dis', pole_dis)
    positive_pole = np.where(pole_dis > 0)[0]
    negative_pole = np.where(pole_dis < 0)[0]
    if negative_pole.size == 0:
        null_count = null_count-1
        # print(pole_dis)
        raise NotImplementedError
    if negative_pole.size == 0:
        null_count = null_count-1
        return np.zeros((1, 1, 52,14))
    positional_embedding = np.zeros((52,14))
    if positive_pole.size == 0:
        negative_pole = [opt_index[idx]-1 for idx in negative_pole]
        negative_pos = pos_top[negative_pole,:]
        negative_pos_middle = negative_pos.copy()
        negative_pos_middle[:,0]+=1
        negative_pos_bot = negative_pos.copy()
        negative_pos_bot[:,0]+=2
        negative_pos = np.concatenate((negative_pos,negative_pos_middle,negative_pos_bot),axis=0)
        rows_pole, cols_pole = zip(*negative_pos)
        positional_embedding[rows_pole, cols_pole]=-1
        potential_dis = np.zeros((52,14))
        for idx_l in range(52):
            for idx_r in range(14):
                neg_dis=np.sum(np.power(negative_pos-np.array([idx_l,idx_r]),2),axis=1)
                if neg_dis.__contains__(0):
                    potential_dis[idx_l,idx_r]=-1
                    continue
                neg_dis = 1/np.sqrt(neg_dis)
                potential_dis[idx_l,idx_r] =-0.15*np.sum(neg_dis) 
    else:
        negative_pole = [opt_index[idx]-1 for idx in negative_pole]
        positive_pole = [opt_index[idx]-1 for idx in positive_pole]
        
        positive_pos = pos_top[positive_pole,:]
        positive_pos_middle = positive_pos.copy()
        positive_pos_middle[:,0]+=1
        positive_pos_bot = positive_pos.copy()
        positive_pos_bot[:,0]+=2
        positive_pos = np.concatenate((positive_pos,positive_pos_middle,positive_pos_bot),axis=0)
        rows_pole, cols_pole = zip(*positive_pos)
        positional_embedding[rows_pole, cols_pole]=1
        negative_pos = pos_top[negative_pole,:]
        negative_pos_middle = negative_pos.copy()
        negative_pos_middle[:,0]+=1
        negative_pos_bot = negative_pos.copy()
        negative_pos_bot[:,0]+=2
        negative_pos = np.concatenate((negative_pos,negative_pos_middle,negative_pos_bot),axis=0)
        rows_pole, cols_pole = zip(*negative_pos)
        positional_embedding[rows_pole, cols_pole]=-1
        potential_dis = np.zeros((52,14))
        r_p = 0.1
    
    
    
        for idx_l in range(52):
            for idx_r in range(14):
                pos_dis=np.sum(np.power(positive_pos-np.array([idx_l,idx_r]),2),axis=1)
                if pos_dis.__contains__(0):
                    potential_dis[idx_l,idx_r]=0
                    continue
            
                neg_dis=np.sum(np.power(negative_pos-np.array([idx_l,idx_r]),2),axis=1)
                if neg_dis.__contains__(0):
                    potential_dis[idx_l,idx_r]=-1
                    continue
                pos_dis = 1/np.sqrt(pos_dis)
                neg_dis = 1/np.sqrt(neg_dis)
                potential_dis[idx_l,idx_r] =-0.2*np.mean(neg_dis)/(np.sum(pos_dis)+np.mean(neg_dis)) 
            

    current = origin_pole_dis[-1]
    pole_dis_new[0][opt_index-1] = pole_dis
    pole_dis_new[0][32] = current 
    # print(pole_dis_new[idx])
    y[0] = potential_dis*current
    x_new[0] = positional_embedding*current


    # print('data generating finished')   
    x_new = np.expand_dims(x_new,1)/10.0
    x_new = x_new.astype('float64')
    y = np.expand_dims(y,1)
    y = (y+5.0)/5.0
    y = y.astype('float64')

    return y

class SCSTask(BaseTask):
    def __init__(
            self, 
            algorithm, 
            dim=17, 
            SI_idx=0,
            latent_dim=16, 
            init_data_path=None, 
            model_path=None,
            bound_path=None,
            eval_latent=False,
            latent_opt=True,
            safe_threshold = 0.05,
            batch_size=1,
            dtype=torch.float32,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ):
        self.SI_idx = SI_idx
        self.bound_path = bound_path
        self.load_dataset()
        self.init_num = 200
        self.evaled_idx = np.zeros(0).astype(np.int32)
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
        


    def load_dataset(self):
        self.dataset = pd.read_excel('./task/scs/data/sel_all.xlsx', index_col=False, header=None).values
        pole_pool = self.dataset[:, 12]
        self.pole_pool = np.array([[float(x) for x in p.split(', ')[:33]] for p in pole_pool])
        self.map_pool = np.load('./task/scs/data/map_all.npz')['map'].reshape(-1, 52*14)
        self.SI = self.dataset[:, :12]
        # print('max_SI', [round(x, 4) for x in np.max(SI, axis=0)[::2]])
        self.safe_dataset = pd.read_excel('task/scs/data/activate.xlsx', index_col=False, header=None).values[:, :6]
    def simulate_fun(self, idx):
        # print(time.time())
        return self.SI[idx, 2*self.SI_idx]

    def safe_fun(self, idx):
        activate = self.safe_dataset[idx]
        return 1-np.max(activate)
    
    def get_fun_vals(self, pole_ids):
        # poles32 = pole16to32(poles)
        util_vals = [self.simulate_fun(x) for x in pole_ids]
        safe_vals = [self.safe_fun(x) for x in pole_ids]
        
        return np.array(util_vals), np.array(safe_vals)

    def load_model(self, model_path):
        '''
        Load model if has model path, or optimize in the original space.
        '''
        assert model_path is not None
        self.model, _ = load_pretrained(
            identifier=model_path[0],
            config_file=model_path[1],
            ckpt_file='model_best.pkl',
            root='./ae_model'
        )
        self.model = self.model.to(dtype=self.dtype, device=self.device)
    
        

    def set_bound(self):
        # Set bounds
        self.lb = np.ones(16)*-1
        # self.lb = np.hstack((self.lb, np.zeros(16)))
        self.lb = np.hstack((self.lb, np.zeros(1)))

        self.ub = np.ones(16)*1
        # self.ub = np.hstack((self.ub, np.zeros(16)))
        self.ub = np.hstack((self.ub, 10*np.ones(1)))
        self.bound = (self.lb, self.ub)

        
        #  Set latent bounds according to initial data , for now
        bound_data = np.load(self.bound_path)
        self.latent_lb = bound_data['lb']
        self.latent_ub = bound_data['ub']
        self.latent_bound = (self.latent_lb, self.latent_ub)

    def load_init_data(self, init_data_path):
        '''
        Load initial data if has data path, or sample inside the bound.
        Transfrom initial data to the latent space if optimize in the latent space.
        '''
        initial_dataset = np.load(init_data_path)
        poles = initial_dataset['poles']
        # print('pole', poles.shape)
        codes = initial_dataset['codes']
        pole_ids = initial_dataset['pole_ids'].astype(np.int32)
        self.evaled_idx = np.hstack((self.evaled_idx, pole_ids))
        # poles, codes, pole_ids = random_pole_generation(20, data_loader.pole_model.module.encoder, pole_pool, pole16=False)
        train_x = codes
        util, safe = self.get_fun_vals(pole_ids)
        self.x = np.hstack((poles[:, :16], poles[:, 32].reshape(-1, 1))) # ndarray (n, dim)

        self.utils = util # ndarray (n, )
        self.safes = safe

        self.latent_x = self.origin_to_latent(self.x)

        if np.all(self.safes<self.safe_threshold):
                print(f'init {self.safes.min()} no safe yet')
        else:
            print(f'init {self.safes.min()} {self.utils[self.safes>self.safe_threshold].max()} {np.max(self.latent_x)}')

    def origin_to_latent(self, x):
        
        all_y = np.zeros((0, 1, 52, 14))
        full_pole = np.zeros((x.shape[0], 33))
        full_pole[:, :16] = x[:, :16]
        full_pole[:, 32] = x[:, 16]
        for xx in full_pole:
            
            pole = np.array([round(p+1)-1 if i<16 else p for i, p in enumerate(xx)])

            if pole.min() >= 0:
                pole[xx[:16].argmin()] = -1
            # print('pole', pole)
            y = pole_data_transfer(pole)
            all_y = np.vstack((all_y, y))
        y_torch = torch.from_numpy(all_y).to(dtype=self.dtype, device=self.device)
        return self.model.encode(y_torch).detach().cpu().numpy()
    
    def latent_to_origin(self, x):
        z = torch.from_numpy(x).to(dtype=self.dtype, device=self.device)
        z_map = self.model.decode(z).reshape(-1, 52*14).detach().cpu().numpy()
        poles = np.zeros((0, 17))
        for mp in z_map:
            map_diff = np.linalg.norm(self.map_pool-mp, axis=1)
            map_diff[self.evaled_idx] = float('inf')
            pole_idx = np.argmin(map_diff)
            full_pole = self.pole_pool[pole_idx]
            pole = np.zeros((1, 17))
            pole[0, :16] = full_pole[:16]
            pole[0, 16] = full_pole[32]
            poles = np.vstack((poles, pole))
            
        return poles
    

    def eval(self, x):
        '''
        Evaluate function value of given x.
        '''
        # print('evall', x)
        if x[16] == 0:
            return 0, 1
        full_pole = np.zeros(33)
        full_pole[:16] = x[:16]
        full_pole[32] = x[16]
        pole_diff = np.linalg.norm(self.pole_pool-full_pole, axis=1)
        pole_diff[self.evaled_idx] = float('inf')
        pole_idx = np.argmin(pole_diff)
        self.evaled_idx = np.hstack((self.evaled_idx, pole_idx))
        util = self.simulate_fun(pole_idx)
        safe = self.safe_fun(pole_idx)
        # print(self.pole_pool[pole_idx])
        # print(util, safe, pole_idx)
        return util, safe