
import os
import time
import argparse
import matplotlib.pyplot as plt
import torch
import gc
from task.gpfun_task import GPTask

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
parser.add_argument('--round', default=100, type=int, help='Round of optimization')
parser.add_argument('--eval_num', default=500, type=int, help='Evaluation number of each round')
parser.add_argument('--dim', default=1000, type=int, help='Original dimension')
parser.add_argument('--latent_dim', default=50, type=int, help='Latent dimension')
parser.add_argument('--fun_dim', default=40, type=int, help='Function dimension')

parser.add_argument('--latent_opt', default=1, type=int, help='0 for original space, 1 for latent space')


args = parser.parse_args()


latent_opt = True if args.latent_opt == 1 else False


print(f'algo {args.algo} {latent_opt}')

# Optimization Loop

exp_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))


for r in range(args.round):
    torch.cuda.empty_cache()
    gc.collect()
    dim = args.dim
    latent_dim = args.latent_dim
    fun_dim = args.fun_dim
    task = GPTask(
        algorithm=args.algo,
        dim=dim,
        latent_dim=latent_dim,
        fun_dim=fun_dim,
        batch_size=10,
        latent_opt=latent_opt,
        )

    best_y = -float('inf')
    while len(task.x) < args.eval_num:
        task.optimize()
        if task.utils[task.safes>task.safe_threshold].max()>best_y:
            print('new best')
            best_y = task.utils[task.safes>task.safe_threshold].max()
        folder = f'{task.algo.save_prefix}_{task.__class__.__name__}_{args.dim}_{args.latent_dim}_{args.fun_dim}_{task.safe_threshold}_{exp_time}_{latent_opt}'
        if not os.path.exists(f'./results/{folder}'):
            os.makedirs(f'./results/{folder}')
        task.save_result(folder, r)
        print(f'Eval {args.algo} {latent_opt} {len(task.utils)}: Best safe Y: {task.utils[task.safes>task.safe_threshold].max()}')

