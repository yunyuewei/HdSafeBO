import numpy as np
import matplotlib.pyplot as plt

# exp_list = [
#     'results/2024-06-03-20-40-43_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/0.npz',
#     'results/2024-06-03-20-40-39_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/0.npz',
#     'results/2024-06-03-21-29-24_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/0.npz',
#     'results/2024-06-03-21-28-45_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/0.npz',
#     'results/2024-06-03-21-57-38_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/0.npz',
#     'results/2024-06-03-21-57-38_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/1.npz',
#     'results/2024-06-03-21-56-58_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/0.npz',
#     'results/2024-06-03-21-56-58_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/1.npz',
#     'results/2024-06-03-23-38-56_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/0.npz',
#     'results/2024-06-03-23-38-50_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/0.npz',
#     'results/2024-06-03-23-38-50_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/1.npz',
#     'results/2024-06-03-23-36-38_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/0.npz',
#     'results/2024-06-03-23-36-38_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/1.npz',
#     'results/2024-06-03-23-36-34_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/0.npz',
#     'results/2024-06-04-00-22-55_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/0.npz',
#     'results/2024-06-04-00-23-01_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/0.npz',
#     'results/2024-06-04-01-12-47_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/0.npz',
#     'results/2024-06-04-01-12-46_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/0.npz',
#     # 'results/2024-06-04-01-19-08_HdSafeBO_2_2_MuscleTask_5_0_act_data2_True/0.npz',
#     'results/2024-06-04-01-40-25_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True/0.npz',
#     'results/2024-06-04-02-06-56_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True',
#     'results/2024-06-04-02-06-58_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True',
#     'results/2024-06-04-02-07-01_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True',
#     'results/2024-06-04-02-07-02_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True',
#     'results/2024-06-04-09-15-30_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True',
#     'results/2024-06-04-09-15-32_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True',
#     'results/2024-06-04-09-15-35_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True',
#     'results/2024-06-04-09-15-37_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True'
#     ]

# Perfromance

# exp_list = [
#     'results/2024-06-04-12-02-47_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True',
#     'results/2024-06-04-12-03-38_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True',
#     'results/2024-06-04-12-03-41_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True',
#     'results/2024-06-04-12-03-43_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True',
#     'results/2024-06-04-15-34-02_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True',
#     'results/2024-06-04-15-34-20_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True',
#     'results/2024-06-04-15-34-23_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True',
#     'results/2024-06-04-15-34-26_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True',
#     'results/2024-06-04-16-45-18_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True',
#     'results/2024-06-04-16-45-33_HdSafeBO_2_2_MuscleTask_6_0_act_data2_True'
# ]


# syn_num = 6

exp_list = [
    # 'results/2024-06-04-15-59-44_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    # 'results/2024-06-04-16-00-02_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    # 'results/2024-06-04-16-44-26_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    # 'results/2024-06-04-16-44-33_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    # 'results/2024-06-04-17-35-05_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    # 'results/2024-06-04-17-35-34_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    # 'results/2024-06-04-19-16-20_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    # 'results/2024-06-04-19-16-46_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    # 'results/2024-06-04-19-23-18_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    # 'results/2024-06-04-19-23-36_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-20-24-10_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-20-24-12_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-20-24-15_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-20-24-17_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-21-20-34_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-21-20-43_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-21-20-45_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-21-20-48_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-22-39-52_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-22-39-55_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-22-39-57_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-22-40-00_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-23-44-19_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-23-44-24_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-23-44-21_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-23-44-27_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-23-44-19_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-23-44-21_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-23-44-24_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-04-23-44-27_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-05-00-41-25_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-05-00-41-32_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-05-00-41-34_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True',
    'results/2024-06-05-00-41-37_HdSafeBO_2_2_MuscleTask_4_0_act_data2_True'
    
]


syn_num = 4





# exp_list = [
#     'results/2024-06-03-20-40-30_HdSafeBO_2_2_MuscleTask_5_0_act_data2_True/0.npz',
#     'results/2024-06-03-20-40-03_HdSafeBO_2_2_MuscleTask_5_0_act_data2_True/0.npz',
#     'results/2024-06-03-20-40-30_HdSafeBO_2_2_MuscleTask_5_0_act_data2_True/0.npz',
#     'results/2024-06-03-20-40-03_HdSafeBO_2_2_MuscleTask_5_0_act_data2_True/0.npz',
#     'results/2024-06-03-21-59-34_HdSafeBO_2_2_MuscleTask_5_0_act_data2_True/0.npz',
#     'results/2024-06-03-21-59-34_HdSafeBO_2_2_MuscleTask_5_0_act_data2_True/1.npz',
#     'results/2024-06-03-21-59-45_HdSafeBO_2_2_MuscleTask_5_0_act_data2_True/0.npz',
#     'results/2024-06-03-21-59-45_HdSafeBO_2_2_MuscleTask_5_0_act_data2_True/1.npz',
#     'results/2024-06-03-21-59-45_HdSafeBO_2_2_MuscleTask_5_0_act_data2_True/2.npz',
#     'results/2024-06-04-00-21-13_HdSafeBO_2_2_MuscleTask_5_0_act_data2_True/0.npz',
#     'results/2024-06-04-00-21-10_HdSafeBO_2_2_MuscleTask_5_0_act_data2_True/0.npz'
#     ]

# syn_num = 5

all_x = np.zeros((0, syn_num*87))
all_utils = np.zeros(0)
all_safes = np.zeros(0)
for exp in exp_list:
    if 'npz' not in exp:
        for i in range(100):
            try:
                data = np.load(f'{exp}/{i}.npz')
                all_x = np.vstack((all_x, data['latent_x'][200:]))
                all_utils = np.hstack((all_utils, data['utils'][200:]))
                all_safes = np.hstack((all_safes, data['safes'][200:]))
                print(f'{exp}/{i}.npz {(data["utils"][200:]).max()} {data["safes"][200:].max()}')
            except:
                continue
    else:
        data = np.load(exp)
        all_x = np.vstack((all_x, data['latent_x'][200:]))
        all_utils = np.hstack((all_utils, data['utils'][200:]))
        all_safes = np.hstack((all_safes, data['safes'][200:]))
        print(f'{exp} {data["utils"][200:].max()} {data["safes"][200:].max()}')

# ind = np.argsort(all_safes)
ind = np.argsort(all_utils)

print(ind[-200:], all_utils[ind[-200:]])



print(all_x.shape, all_utils.shape, all_safes.shape)

plt.hist(all_utils[ind[-200:]])
plt.savefig(f'util_{syn_num}.png')

np.savez(f'task/muscle/init_data/init_{syn_num*87}_act_data2_leg_sort.npz', x=all_x[ind[-200:]], r=all_utils[ind[-200:]], v=all_safes[ind[-200:]])



bound_path=f'task/muscle/SB3/logs/Leg-v2/0531-095220_42/checkpoint/bound_{syn_num}_0_100_act_data2.npz'

bound_data = np.load(bound_path)
lb_act = bound_data['lb']
ub_act = bound_data['ub']
print(lb_act, ub_act)

lb = np.zeros(0)
ub = np.zeros(0)
for i in range(syn_num):
    lb = np.hstack((lb, lb_act[i] * np.ones(87)))
    ub = np.hstack((ub, ub_act[i] * np.ones(87)))
    
top_x = all_x[ind[-200:]]

ratio = (top_x.max(0)-top_x.min(0))/(ub-lb)
print(top_x.max(0), top_x.min(0))


print(ratio, ratio.max(), ratio.min(), ratio.mean(), ratio.std())

np.savez(f'task/muscle/init_data/bound_{syn_num*87}_act_data2_leg_sort.npz', latent_lb=top_x.min(0), latent_ub=top_x.max(0))

