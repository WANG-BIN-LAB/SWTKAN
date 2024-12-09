import sys
sys.path.append("/home/cxc/code/TOSICA-2layer-2/")
import anndata as ad
import TOSICA
import scanpy as sc
import numpy as np
import warnings
warnings.filterwarnings ("ignore")

import torch.nn as nn
import torch

print(torch.__version__)
print(torch.cuda.get_device_capability(device=None),  torch.cuda.get_device_name(device=None))


# read h5ad data immune
immune_train = sc.read_h5ad('/data1/home/chenxc/data/BBenchmarking/immune/immune_train.h5ad')
immune_test = sc.read_h5ad('/data1/home/chenxc/data/BBenchmarking/immune/immune_test.h5ad')

epochs = 10
project = 'immune_2layer_128token_4window'
print('training...............')
TOSICA.train(immune_train, label_name='celltype', epochs=epochs, project=project, batch_size=8, max_gs=128, window_size=4, embed_dim=24)

print('predict...............')
for i in range(epochs):
    model_weight_path = '/data1/home/chenxc/code/TOSICA-2layer-2/test/immune_2layer_128token_4window/model-%d.pth' % (i)
    new_adata = TOSICA.pre(immune_test, model_weight_path=model_weight_path, project=project, batch_size=8, max_gs=128, window_size=4, embed_dim=24)
    correct_sum = 0
    Prediction = new_adata.obs['Prediction']
    cell = immune_test.obs['celltype']
    if i == epochs-1:
        new_adata.write('/data1/home/chenxc/code/TOSICA-2layer-2/test/immune_2layer_128token_4window/pre_adata.h5ad')
    else:
        del immune_test.obs['Prediction']
        del immune_test.obs['Probability']
    for j in range(new_adata.n_obs):
        p = Prediction.get(j)
        c = cell.get(j)
        if c == p:
            correct_sum=correct_sum+1
    print(str(i) + " " + str(correct_sum / new_adata.n_obs))