
import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import gc

# from task.muscle_task import MuscleTask
from task.muscle.task_env_mjhand import MuscleSynergyEnv
import warnings
import botorch
from botorch.test_functions import SyntheticTestFunction,Levy, Shekel, Michalewicz
from botorch.utils.transforms import unnormalize
from botorch.exceptions import BotorchWarning, InputDataWarning



exp = 'act_data2'
syn_num = 6
env = MuscleSynergyEnv(syn_num=syn_num, 
                            #  model_path='task/muscle/SB3/logs/Leg-v2/0531-095220_42/checkpoint/pca_5.pkl',
                             model_path=f'task/muscle/SB3/logs/Leg-v2/0531-095220_42/checkpoint/pca_{syn_num}_{exp}.pkl',
                             )
dim = env.dim

# bound_path = f'task/muscle/SB3/logs/Leg-v2/0531-095220_42/checkpoint/bound_5_10_100.npz'

bound_path=f'task/muscle/SB3/logs/Leg-v2/0531-095220_42/checkpoint/bound_{syn_num}_0_100_{exp}.npz'

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


data = np.load('task/muscle/init_data/init_522_act_data2_leg_safe_sort.npz')

syn = data['x']

# print(data['v'].min(), data['v'].max())
all_r = np.zeros(0)
all_v = np.zeros(0)
for i, s in enumerate(syn):
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
    print(len(all_r), r, v, data['v'][i])
    # break

np.savez(f'./task/muscle/init_data/init_{dim}_{exp}_{env.env_type}_sort.npz', x=syn, r=all_r, v=all_v)


