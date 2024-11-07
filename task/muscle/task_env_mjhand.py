import os 
import sys 
os.environ["GYM_ENV"] = "gym"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './mj_HandHold')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ae_model')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../autoencoder')))

from models import load_pretrained

import math
import torch
import msmodel_gym

import joblib
from task.muscle.muscle_utils import *
from gym.wrappers import RecordVideo



# syn_num = 6

# Simple solver to compute landing z-axis velocity
def solve_quadratic_equation(a, b, c):
    delta = b**2 - 4*a*c
    # if delta < 0:
    #     return NotImplementedError
    assert delta >= 0
    if delta == 0:
        x = -b / (2*a)
        return x
    else:
        x1 = (-b + math.sqrt(delta)) / (2*a)
        x2 = (-b - math.sqrt(delta)) / (2*a)
        # print(f'solution {x1} {x2}')
        # if x1*x2 > 0:
        #     raise NotImplementedError
        # else:
        assert x1*x2 < 0
        return max(x1, x2)


class MuscleSynergyEnv:
    def __init__(self, syn_num, model_path,latent_opt=True, mode='Linear',root='./', env_type='leg', dtype=torch.float, device=torch.device('cuda')):
        
        self.syn_num = syn_num
        self.mode = mode
        self.dtype = dtype
        self.device = device
        self.root = root
        self.latent_opt = latent_opt
        self.env_type = env_type
        if self.env_type == 'hand':
            self.origin_env = gym.make('msmodel_gym/HandFloat-v1')
        else:
            self.origin_env = gym.make('msmodel_gym/Leg-v2')
        if mode == 'Linear':
            self.pca = self.load_syn_model(model_path)
            self.env = SynergyWrapper(self.origin_env, self.pca)
        else:
            self.model = self.load_ae_model(model_path).to(dtype=self.dtype, device=self.device)
            self.env = SynergyAEWrapper(self.origin_env, self.model, self.dtype, self.device, syn_num=syn_num)
        # self.env._get_obs()
        # self.obs_shape = self.get_obs_without_action().shape[0] # Not include action and time
        self.obs_shape = self.env.observation_space.shape[0] - self.origin_env.action_space.shape[0] -1 # Not include action and time

        # print(f'obs_shape {self.obs_shape}')
        # raise NotImplementedError
        self.act_shape = self.env.action_space.shape[0]
        self.origin_act_shape = self.origin_env.action_space.shape[0]
        self.dim = self.obs_shape * self.act_shape
        self.origin_dim = self.obs_shape * self.origin_act_shape
        if self.latent_opt:
            self.weight_matrix = self.build_weight_matrix((self.act_shape, self.obs_shape))
        else:
            self.weight_matrix = self.build_weight_matrix((self.origin_act_shape, self.obs_shape))

        self.lb = -3 * np.ones(self.dim)
        self.ub = 3 * np.ones(self.dim)
        
        # print(syn_num, model_path,latent_opt, mode,root, env_type)
        # raise NotImplementedError
        

    def load_syn_model(self, pca_path):
        pca = joblib.load(pca_path)
        return pca
    
    def set_bound(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def load_ae_model(self, model_path):
        identifier, config, ckpt = model_path
        model, cfg = load_pretrained(
            identifier=identifier,
            config_file=config,
            ckpt_file=ckpt,
            root=f'{self.root}ae_model'
            )
        return model
    
    def get_obs_without_action(self):
        obs = np.zeros(0)
        obs_dict = self.env.obs_dict
        # print([(key, len(obs_dict[key])) for key in obs_dict.keys()])
        for key in obs_dict.keys():
            if key == 'act' or key == 'time' or 'muscle' in key:
                continue
            # print(key, obs_dict[key].shape)
            obs = np.hstack((obs, obs_dict[key]))
        # print(obs.shape)
        # raise NotImplementedError
        return obs


    def reset(self):
        # return self.env.reset(seed=0)
        if self.latent_opt:
            return self.env.reset() # noise
        else:
            return self.origin_env.reset()

    def step(self, action):
        if self.latent_opt:
            return self.env.step(action)
        else:
            return self.origin_env.step(action)

    def build_weight_matrix(self, shape):
        return np.random.randn(*shape)

    def get_action(self, obs):
        return np.dot(self.weight_matrix, obs.T)

    def update_weight_matrix(self, updated_weight_matrix):
        if updated_weight_matrix.shape != self.weight_matrix.shape:
            updated_weight_matrix = updated_weight_matrix.reshape(
                self.weight_matrix.shape)
        self.weight_matrix = updated_weight_matrix
    
    def syn_to_mus_policy(self, syn_policy):
        # syn_policy: ndarray: n * env.dim
        # syn_policy_line: ndarray: (n*obs_shape) * act_Shape
        assert syn_policy.ndim == 2
        syn_policy_line = syn_policy.reshape(syn_policy.shape[0]*self.obs_shape, self.act_shape)
        if self.mode == 'Linear':
            mus_policy_line = self.pca.inverse_transform(syn_policy_line)
        else:
            arr = torch.from_numpy(syn_policy_line).to(dtype=self.dtype, device=self.device)
            mus_policy_line = self.model.decode(arr).detach().cpu().numpy()
        return mus_policy_line.reshape(syn_policy.shape[0], self.origin_dim)
    

    def mus_to_syn_policy(self, mus_policy):
        assert mus_policy.ndim == 2
        mus_policy_line = mus_policy.reshape(mus_policy.shape[0]*self.obs_shape, self.origin_act_shape)
        if self.mode == 'Linear':
            syn_policy_line = self.pca.transform(mus_policy_line)
        else:
            arr = torch.from_numpy(mus_policy_line).to(dtype=self.dtype, device=self.device)
            syn_policy_line = self.model.encode(arr).detach().cpu().numpy()
        return syn_policy_line.reshape(mus_policy.shape[0], self.dim)

    def __call__(self, updated_weight_matrix):
        # print(updated_weight_matrix)
        # data = np.load('results/2024-06-03-12-54-47_HdSafeBO_2_2_MuscleTask_5_0_act_data4_True/0.npz')
       
        # policy = data['latent_x'][200].reshape(1, -1)
        # print(policy-updated_weight_matrix)
        # print(f'max_val {updated_weight_matrix.max()} {updated_weight_matrix.min()}')
        # print(self.lb.shape, self.ub.shape)
        
        updated_weight_matrix = np.clip(updated_weight_matrix, self.lb, self.ub)
        assert np.all(updated_weight_matrix <= self.ub) and np.all(
            updated_weight_matrix >= self.lb), [
                updated_weight_matrix.max(), 
                updated_weight_matrix.min(), 
                self.lb[updated_weight_matrix.argmin()],
                self.ub[updated_weight_matrix.argmax()]]
        self.update_weight_matrix(updated_weight_matrix)  

        # obs = self.reset()
        full_obs = self.reset()
        obs = self.get_obs_without_action()

        totalReward = 0
        done = False
        total_step = 0
        while not done:
            
            action = self.get_action(obs).T
            full_obs, reward, terminated, truncated, info = self.step(action)
            obs = self.get_obs_without_action()
            totalReward += reward
            total_step += 1
            done = terminated or truncated
            # velocity = self.env.data.qvel.flat.copy()[-6]
            if self.env_type == 'hand':
                velocity = np.linalg.norm(self.env.data.qvel.flat.copy()[-6:-3])
                pos = self.env.data.body('waterbottle').xpos.flat.copy()
        if self.env_type == 'hand':
            # print('velo', velocity, pos, totalReward)
            # v_land = 0
            # if terminated: # bottle drop
            v_land = -velocity
            return totalReward, max(v_land, -10)
        else:
            # print(totalReward, total_step)
            # raise NotImplementedError
            return totalReward, total_step
            # return total_step, total_step
            # print('drop')
        # print(totalReward)
        # else:
        #     print(f'no drop {v_land} {totalReward}')
        
    
    def record_video(self, updated_weight_matrix):
        updated_weight_matrix = np.clip(updated_weight_matrix, self.lb, self.ub)
        assert np.all(updated_weight_matrix <= self.ub) and np.all(
            updated_weight_matrix >= self.lb), [
                updated_weight_matrix.max(), 
                updated_weight_matrix.min(), 
                self.lb[updated_weight_matrix.argmin()],
                self.ub[updated_weight_matrix.argmax()]]
        self.update_weight_matrix(updated_weight_matrix)
        self.env = RecordVideo(self.env, video_folder='./functions/muscle',name_prefix="test-video", episode_trigger=lambda x: x % 2 == 0)
        # obs = self.reset()
        full_obs = self.reset()
        
        obs = self.get_obs_without_action()
        self.env.start_video_recorder()
        totalReward = 0
        for i in range(1000):
            print(i)
            
            # done = False
            # while not done:
            
            action = self.get_action(obs).T
            full_obs, reward, terminated, truncated, info = self.step(action)
            # self.env.render()
            obs = self.get_obs_without_action()
            totalReward += reward
            done = terminated or truncated
            if done:
                print(totalReward)
                # raise NotImplementedError
                totalReward = 0
                full_obs = self.reset()
                obs = self.get_obs_without_action()
                break
        self.env.close_video_recorder()
        self.env.close



def generate_init_data():
    
    for syn_num in [4]:
        exp = 'act_data2'
        env = MuscleSynergyEnv(syn_num, 
                            # f'./pca/pca_{syn_num}_{exp}.pkl',
                        f'./SB3/logs/Leg-v2/0531-095220_42/checkpoint/pca_{syn_num}_{exp}.pkl',
                        root='../../',
                        mode='Linear',
                        
                        )
        dim = syn_num * env.obs_shape
        print('dim', dim)
        safe = False
        x = np.zeros(env.dim)
        for i in range(1):
            r, v0 = env(x)
            print(r, v0)
        bound_path = f'./SB3/logs/Leg-v2/0531-095220_42/checkpoint/bound_{syn_num}_0_100_{exp}.npz'
        bound_data = np.load(bound_path)
        lb_act = bound_data['lb']
        ub_act = bound_data['ub']
        print(lb_act, ub_act)

        lb = np.zeros(0)
        ub = np.zeros(0)
        for i in range(env.act_shape):
            lb = np.hstack((lb, lb_act[i] * np.ones(env.obs_shape)))
            ub = np.hstack((ub, ub_act[i] * np.ones(env.obs_shape)))
        env.set_bound(lb, ub)
        while(True):
            print('resample')
            syn = np.random.random((200, dim))
            syn = lb + syn * (ub - lb)
            all_r = np.zeros(0)
            all_v = np.zeros(0)
            for s in syn:
                # print(s)
                # data = np.load('../../results/2024-06-03-19-18-44_HdSafeBO_2_2_MuscleTask_5_0_act_data2_True/0.npz')

                # policy = data['latent_x'][0]
                # print(policy-s)
                # raise NotImplementedError
                r, v = env(s.reshape(1, -1))
                print(syn_num, r, v)
                if v>-4.2:
                    safe = True
                all_r = np.hstack((all_r, r))
                all_v = np.hstack((all_v, v))
                print(len(all_r), r, v)
                
            if safe:
                break
        print(f'generate {syn_num} {dim}')
        np.savez(f'./init_data/init_{dim}_{exp}_{env.env_type}.npz', x=syn, r=all_r, v=all_v)

def test_policy():
    pass

if __name__ == "__main__":
    np.random.seed(0)
    generate_init_data()
    # syn_num = 5

    # env = MuscleSynergyEnv(syn_num, 
    #                      [f'Muscle/DIM_{syn_num}/irvae_muscle',
    #                     f'muscle_irvae_z{syn_num}.yml',
    #                     f'model_best.pkl'],
    #                     root='../../',
    #                     mode='AE',
    #                     latent_opt=True
    #                     )
    # print(env.origin_env.action_space)
    # raise NotImplementedError
    # data = np.load('../../results/OLSSO_2_2_MuscleTask_-5.29_2023-09-05-23-37-41/0.npz')

    # # util = data['utils']
    # # print(f'max {util.max()}')
    # # x = data['latent_x']
    # # policy = x[util.argmax()]
    # # r, v = env(policy)
    # # print(r, v)

    # lb_act = bound_data['lb']
    # ub_act = bound_data['ub']

    # # mean_act = bound_data['mean']
    # # std_act = bound_data['std']

    # # lb_act = mean_act - 2*std_act
    # # ub_act = mean_act + 2*std_act
    # print(lb_act, ub_act)

    # lb = np.zeros(0)
    # ub = np.zeros(0)
    # for i in range(self.model.act_shape):
    #     lb = np.hstack((lb, lb_act[i] * np.ones(self.model.obs_shape)))
    #     ub = np.hstack((ub, ub_act[i] * np.ones(self.model.obs_shape)))
    # self.latent_lb = lb
    # self.latent_ub = ub

    # bound_data = np.load(f'./init_data/bound_{syn_num}_irvae_muscle.npz')
    # lb_act = bound_data['lb']
    # ub_act = bound_data['ub']
    # print(lb_act, ub_act)
    # lb = np.zeros(0)
    # ub = np.zeros(0)
    # for i in range(env.act_shape):
    #     lb = np.hstack((lb, lb_act[i] * np.ones(env.obs_shape)))
    #     ub = np.hstack((ub, ub_act[i] * np.ones(env.obs_shape)))
    
    # max_act = np.ones((1, env.origin_act_shape*env.obs_shape))
    # min_act = np.zeros((1, env.origin_act_shape*env.obs_shape))

    # lb = env.mus_to_syn_policy(min_act).flatten()
    # ub = env.mus_to_syn_policy(max_act).flatten()
    # print(lb)
    # print(ub)
    # env.set_bound(lb, ub)

    # data = np.load('../../results/OLSSO_2_2_MuscleTask_5_-3.2_2023-09-28-16-29-52_irvae_muscle/0.npz')
    # policy = data['latent_x']
    # utils = data['utils']
    # safes=data['safes']

    # # data = np.load('./init_data/init_325_irvae_muscle.npz')
    # # policy = data['x']
    # # utils = data['r']
    # # safes=data['v']

    # print(utils[0], utils.max(), len(policy), len(utils))
    # # print(laten_x[200:210])
    # # best_policy = np.array(policy[np.argmax(utils)])

    # # f = env(best_policy)
    # for i, p in enumerate(policy[200:]):
    #     p = np.array(p)
    #     f = env(p)

    #     print(i, f, utils[200+i], safes[200+i])

    # print(f)


