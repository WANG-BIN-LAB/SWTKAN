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


# read h5ad data lung
lung_train = sc.read_h5ad('/home/cxc/data/scRNA-seq/Experiment/human_lung/lung_train.h5ad')
lung_test = sc.read_h5ad('/home/cxc/data/scRNA-seq/Experiment/human_lung/lung_test.h5ad')

epochs = 10
project = 'lung'
print('training...............')
TOSICA.train(lung_train, label_name='Celltype', epochs=epochs, project=project, batch_size=8, max_gs=128, window_size=4, embed_dim=24)

print('predict...............')
for i in range(epochs):
    model_weight_path = '/home/cxc/code/TOSICA-2layer-2/test/lung/model-%d.pth' % (i)
    new_adata = TOSICA.pre(lung_test, model_weight_path=model_weight_path, project=project, batch_size=8, max_gs=128, window_size=4, embed_dim=24)
    correct_sum = 0
    Prediction = new_adata.obs['Prediction']
    cell = lung_test.obs['Celltype']
    if i == epochs-1:
        new_adata.write('/home/cxc/code/TOSICA-2layer-2/test/lung/pre_adata.h5ad')
    else:
        del lung_test.obs['Prediction']
        del lung_test.obs['Probability']
    for j in range(new_adata.n_obs):
        p = Prediction.get(j)
        c = cell.get(j)
        if c == p:
            correct_sum=correct_sum+1
    print(str(i) + " " + str(correct_sum / new_adata.n_obs))