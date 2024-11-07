
import os
import time
import argparse
import matplotlib.pyplot as plt
import torch
import gc
from task.digit_task import DigitTask

import warnings
import botorch
from botorch.exceptions import BotorchWarning, InputDataWarning

suppress = True
warnings.simplefilter("ignore" if suppress else "default", BotorchWarning)
warnings.simplefilter("ignore" if suppress else "default", InputDataWarning)
warnings.simplefilter("ignore" if suppress else "default", FutureWarning)
warnings.simplefilter("ignore" if suppress else "default", UserWarning)



parser = argparse.ArgumentParser(description='BO experiment')

parser.add_argument('--algo', default='HdSafeBO', type=str, help='Optimizer name')
parser.add_argument('--round', default=20, type=int, help='Round of optimization')
parser.add_argument('--eval_num', default=200, type=int, help='Evaluation number of each round')
parser.add_argument('--task', default='Digit', type=str, help='Optimization task name')
parser.add_argument('--latent_dim', default=6, type=int, help='Latent dimension')
parser.add_argument('--ending', default='1919', type=str, help='Model ending number')
parser.add_argument('--latent_opt', default=1, type=int, help='0 for original space, 1 for latent space')


args = parser.parse_args()


latent_opt = True if args.latent_opt == 1 else False


print(f'algo {args.algo} {latent_opt}')

# Optimization Loop

exp_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
task_id = range(10)
for idx in task_id:

    for r in range(args.round):
        torch.cuda.empty_cache()
        gc.collect()
       
        latent_dim = args.latent_dim
        idt = 'MNIST'
        exp = f'20230922-{args.ending}'
       
        identifier=f'{idt}/DIM_{latent_dim}/{exp}'
        config_file=f'mnist_irvae_z{latent_dim}.yml'
        bound_path = f'./task/digit/init_data/bound_{latent_dim}_20230922-{args.ending}.npz'
        # task_idx = int(args.task.split('_')[-1])
        task = DigitTask(
            algorithm=args.algo,
            latent_dim=latent_dim,
            task_idx=[idx],
            batch_size=1,
            model_path=[[identifier, config_file],
                        './task/digit/model/mnist_cnn.pt'],  # List, VAE model and score model
            bound_path = bound_path,
            latent_opt=latent_opt,
            init_data_path='./task/digit/mnist_data/'
            )
        
        best_y = -float('inf')
        while len(task.x) < args.eval_num:
            task.optimize()
            if task.utils[task.safes>task.safe_threshold].max()>best_y:
                print('new best')
                best_y = task.utils[task.safes>task.safe_threshold].max()
            folder = f'{task.algo.save_prefix}_{task.__class__.__name__}_{idx}_{args.latent_dim}_{args.latent_dim}_{task.safe_threshold}_{exp_time}_{latent_opt}'
            if not os.path.exists(f'./results/{folder}'):
                os.makedirs(f'./results/{folder}')
            task.save_result(folder, r)
            print(f'Eval {args.algo} {latent_opt} {idx} {len(task.utils)}: Best safe Y: {task.utils[task.safes>task.safe_threshold].max()}')

