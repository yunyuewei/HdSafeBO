
import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import gc

from task.muscle_task import MuscleTask

import warnings
import botorch
from botorch.exceptions import BotorchWarning, InputDataWarning

suppress = True
warnings.simplefilter("ignore" if suppress else "default", BotorchWarning)
warnings.simplefilter("ignore" if suppress else "default", InputDataWarning)
warnings.simplefilter("ignore" if suppress else "default", FutureWarning)
warnings.simplefilter("ignore" if suppress else "default", UserWarning)

parser = argparse.ArgumentParser(description='BO experiment')

parser = argparse.ArgumentParser(description='BO experiment')

parser.add_argument('--algo', default='HdSafeBO', type=str, help='Optimizer name')
parser.add_argument('--round', default=50, type=int, help='Round of optimization')
parser.add_argument('--eval_num', default=1000, type=int, help='Evaluation number of each round')
parser.add_argument('--syn_num', default=5, type=int, help='Optimization task name')
parser.add_argument('--latent_opt', default=1, type=int, help='0 for original space, 1 for latent space')

args = parser.parse_args()
latent_opt = True if args.latent_opt == 1 else False

print(args)
print(f'algo {args.algo} {args.latent_opt}')
os.environ["GYM_ENV"] = "gym"
# Optimization Loop
# CUDA_VISIBLE_DEVICES=0 GYM_ENV=gym python optimization_muscle.py --algo SCBO

exp_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

for r in range(args.round):
    try:
        torch.cuda.empty_cache()
        gc.collect()
        latent_dim = args.syn_num*65
        # task = MuscleTask(
        #     algorithm=args.algo,
        #     batch_size=10,
        #     model_path=[
        #         f'task/muscle/syn_data/mjhand_2/ica_{args.syn_num}.pkl',
        #         f'task/muscle/syn_data/mjhand_2/pca_{args.syn_num}.pkl',
        #         f'task/muscle/syn_data/mjhand_2/normalizer_{args.syn_num}.pkl',
        #     ],
        #     init_data_path=f'task/muscle/init_data/init_{latent_dim}.npz',
        #     syn_num=args.syn_num,
        #     mode='Linear'
        # )
        exp = 'act_data4'
        task = MuscleTask(
            algorithm=args.algo,
            batch_size=10,
            dim = 55*65,
            latent_dim=latent_dim,
            model_path=f'task/muscle/pca/pca_{args.syn_num}_{exp}.pkl',
            bound_path= None,
            init_data_path=f'task/muscle/init_data/init_{latent_dim}_{exp}.npz',
            syn_num=args.syn_num,
            latent_opt=latent_opt,
            eval_latent=latent_opt,
            mode='Linear'
        )


        best_y = -float('inf')
        while len(task.x) < args.eval_num:
            task.optimize()
            if np.all(task.safes<task.safe_threshold):
                print(f'{args.algo} Eval {len(task.utils)}: No safe yet')
            else:
                if task.utils[task.safes>task.safe_threshold].max()>best_y:
                    print('new best')
                    best_y = task.utils[task.safes>task.safe_threshold].max()
                folder = f'{task.algo.save_prefix}_{task.__class__.__name__}_{task.syn_num}_{task.safe_threshold}_{exp_time}_{exp}_{latent_opt}'
                if not os.path.exists(f'./results/{folder}'):
                    os.makedirs(f'./results/{folder}')
                task.save_result(folder, r)
                print(f'{args.algo} Eval {len(task.utils)}: Best safe Y: {task.utils[task.safes>task.safe_threshold].max()}')
    except botorch.exceptions.errors.ModelFittingError: # just continue if we reach this error
        continue