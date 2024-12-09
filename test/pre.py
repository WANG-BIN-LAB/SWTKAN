import scanpy as sc
import gc
import pandas as pd
import numpy as np

adata = sc.read_h5ad('/data1/home/chenxc/data/BBenchmarking/immune/Immune_ALL_human.h5ad')
for i in adata.obs.columns:
    if i != 'cell_type':
        del adata.obs[i]
del adata.layers
gc.collect()

arr = np.arange(adata.n_obs)
np.random.seed(0)
np.random.shuffle(arr)
train_adata = adata[arr[:int(adata.n_obs*0.9)]]
test_adata = adata[arr[int(adata.n_obs*0.9):]]

adata.write('/data1/home/chenxc/data/BBenchmarking/immune/immune.h5ad')
train_adata.write('/data1/home/chenxc/data/BBenchmarking/immune/immune_train.h5ad')
test_adata.write('/data1/home/chenxc/data/BBenchmarking/immune/immune_test.h5ad')