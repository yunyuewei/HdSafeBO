import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib


data_path = 'act_data4'
acts = np.load(f'{data_path}.npy')
syn_dict = {}
for i in range(10):
    print(i)
    pca = PCA(n_components=i + 1)
    _ = pca.fit_transform(acts)
    syn_dict[i + 1] = round(sum(pca.explained_variance_ratio_), 4)
    joblib.dump(pca, f'pca_{i+1}_{data_path}.pkl')
    


plt.plot(list(syn_dict.keys()), list(syn_dict.values()))
plt.title('VAF by N synergies')
plt.xlabel('# synergies')
plt.ylabel('VAF')
plt.grid()
# plt.show()
plt.savefig(f"./sub_synergies_{data_path}.png")

    