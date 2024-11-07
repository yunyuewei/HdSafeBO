
import os
import time
import argparse
import matplotlib.pyplot as plt
import torch
import gc
from task.scs_task import SCSTask
from task.digit_task import DigitTask
import warnings
suppress = True
from botorch.exceptions import BotorchWarning, InputDataWarning

warnings.simplefilter("ignore" if suppress else "default", BotorchWarning)
warnings.simplefilter("ignore" if suppress else "default", InputDataWarning)
warnings.simplefilter("ignore" if suppress else "default", FutureWarning)
warnings.simplefilter("ignore" if suppress else "default", UserWarning)

parser = argparse.ArgumentParser(description='BO experiment')

parser.add_argument('--algo', default='HdSafeBO', type=str, help='Optimizer name')
parser.add_argument('--round', default=10, type=int, help='Round of optimization')
parser.add_argument('--eval_num', default=1000, type=int, help='Evaluation number of each round')
parser.add_argument('--task', default=0, type=int, help='Optimization task name')
parser.add_argument('--latent_opt', default=1, type=int, help='0 for original space, 1 for latent space')


args = parser.parse_args()


latent_opt = True if args.latent_opt == 1 else False


print(f'algo {args.algo} {latent_opt}')

# Optimization Loop

exp_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
task_id = range(10)
# for SI_idx in [0, 1, 2]:
for SI_idx in [4]:
    for r in range(args.round):
        torch.cuda.empty_cache()
        gc.collect()
        
        latent_dim = 16
        idt = 'SCS'
        exp = 'irvae_scs'

        # latent_dim = 10
        # idt = 'SCS'
        # exp = '20230921-2027'

        # latent_dim = 6
        # idt = 'SCS'
        # exp = '20230921-1953'
        
        identifier=f'{idt}/DIM_{latent_dim}/{exp}'
        config_file=f'scs_irvae_z{latent_dim}.yml'
        
        bound_path = f'task/scs/data/bound_{latent_dim}_{idt}_{exp}_218000.npz'
        # task_idx = int(args.task.split('_')[-1])
        task = SCSTask(
            algorithm=args.algo,
            latent_dim=latent_dim,
            SI_idx=SI_idx,
            batch_size=10,
            model_path=[identifier, config_file],
            bound_path=bound_path,
            latent_opt=latent_opt,
            init_data_path=f'./task/scs/data/initial_200/{SI_idx}/{0}.npz',
            )
        print('safe threshold', task.safe_threshold)
        
        best_y = -float('inf')
        while len(task.x) < args.eval_num:
            task.optimize()
            if task.utils[task.safes>task.safe_threshold].max()>best_y:
                print('new best')
                best_y = task.utils[task.safes>task.safe_threshold].max()
            folder = f'{exp_time}_{task.algo.save_prefix}_{task.__class__.__name__}_{SI_idx}_{task.safe_threshold}_{latent_dim}_{latent_opt}'
            if not os.path.exists(f'./results/{folder}'):
                os.mkdir(f'./results/{folder}')
            task.save_result(folder, r)
            print(f'Eval {args.algo} {latent_opt} {SI_idx} {len(task.utils)}: Best safe Y: {task.utils[task.safes>task.safe_threshold].max()}')

