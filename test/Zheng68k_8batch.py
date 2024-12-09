import sys
sys.path.append("/home/cxc/code/TOSICA-2layer-2/")
import anndata as ad
import TOSICA
import scanpy as sc
import numpy as np
import warnings
warnings.filterwarnings ("ignore")

import torch

print(torch.__version__)
print(torch.cuda.get_device_capability(device=None),  torch.cuda.get_device_name(device=None))



def get_KFold_data(k, i, data):
    # get the training and validation sets for the i-th fold in k-fold cross validation
    # data is an object of AnnData
    # k is the number of folds, i belongs to {0, 1, 2, ... , k-1}
    fold_size = data.n_obs // k  # the size of each fold
    val_start = i * fold_size  # initial index of validation set
    if i>=0 and i<=k-1 and isinstance(i, int) :
        val_end = (i+1)*fold_size  # the end index of validation set
        valid_data = data[val_start:val_end, :]
        train_data = ad.concat([data[:val_start, :], data[val_end:, :]])
        return train_data, valid_data
    else:
        print('Error!!! i belongs to {0, 1, 2, ... , k-1}')
        return ad.AnnData(), ad.AnnData()




# ref_adata = sc.read('/home/cxc/code/TOSICA-1layer/data/demo_train.h5ad')
# ref_adata = ref_adata[:,ref_adata.var_names]
# query_adata = sc.read('/home/cxc/code/TOSICA-1layer/data/demo_test.h5ad')
# query_adata = query_adata[:,ref_adata.var_names]

train_adata = sc.read_h5ad('/data1/home/chenxc/data/Zheng68K/Zheng68K_train.h5ad')
test_adata = sc.read_h5ad('/data1/home/chenxc/data/Zheng68K/Zheng68K_test.h5ad')

epochs = 50
project = 'Zheng68k_2layer_128token_4window'

TOSICA.train(train_adata, label_name='celltype',epochs=epochs,project=project, batch_size=8, max_gs=128,window_size=4,embed_dim=24)


for i in range(epochs):
    model_weight_path = '/data1/home/chenxc/code/TOSICA-2layer-2/test/Zheng68k_2layer_128token_4window/model-%d.pth' %(i)
    new_adata = TOSICA.pre(test_adata, model_weight_path=model_weight_path, project=project,batch_size=8, max_gs=128,window_size=4,embed_dim=24)
    correct_sum = 0
    Prediction = new_adata.obs['Prediction']
    cell = test_adata.obs['celltype']
    if i == epochs - 1:
        new_adata.write('/data1/home/chenxc/code/TOSICA-2layer-2/test/Zheng68k_2layer_128token_4window/pre_adata.h5ad')
    else:
        del test_adata.obs['Prediction']
        del test_adata.obs['Probability']
    for j in range(new_adata.n_obs):
        p = Prediction.get(j)
        c = cell.get(j)
        if c == p:
            correct_sum = correct_sum + 1
    print(str(i) + " " + str(correct_sum / new_adata.n_obs))

# for i in range(len(correct)):
#     print(str(i) + " " + str(correct[i]))

# new_adata = TOSICA.pre(test_adata, model_weight_path=model_weight_path, project='Zheng68K_swin_trans_1_layer_8_blocks_4batch')

# model_weight_path = '/home/cxc/code/TOSICA-main/test/hGOBP_demo_10/model-99.pth'
# new_adata = TOSICA.pre(query_adata, model_weight_path = model_weight_path,project='hGOBP_demo_10')
#
# #new_adata
# new_adata.write('demo_data.h5ad')
#
# correct_sum=0
# Prediction=new_adata.obs['Prediction']
# cell=query_adata.obs['Celltype']
# for i in range(new_adata.n_obs):
#     p=Prediction.get(i)
#     c=cell.get(i)
#     if c==p:
#         correct_sum=correct_sum+1
#
# print('correct : ' + str(correct_sum/new_adata.n_obs))






#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# num_genes = adata.shape[1]
# #mask_path = os.getcwd()+project+'/mask.npy'
# mask = np.load(mask_path)
# project_path = os.getcwd()+'/%s'%project
# pathway = pd.read_csv(project_path+'/pathway.csv', index_col=0)
# dictionary = pd.read_table(project_path+'/label_dictionary.csv', sep=',',header=0,index_col=0)
# n_c = len(dictionary)
# label_name = dictionary.columns[0]
# dictionary.loc[(dictionary.shape[0])] = 'Unknown'
# dic = {}
# for i in range(len(dictionary)):
#     dic[i] = dictionary[label_name][i]
# model = create_model(num_classes=n_c, num_genes=num_genes,mask = mask, has_logits=False,depth=depth,num_heads=num_heads).to(device)
# # load model weights
# model.load_state_dict(torch.load(model_weight_path, map_location=device))
# model.eval()
# parm={}
# for name,parameters in model.named_parameters():
#     #print(name,':',parameters.size())
#     parm[name]=parameters.detach().cpu().numpy()
#
#
# latent = torch.empty([0,embed_dim]).cpu()
# att = torch.empty([0,(len(pathway))]).cpu()
# predict_class = np.empty(shape=0)
# pre_class = np.empty(shape=0)
# latent = torch.squeeze(latent).cpu().numpy()
# l_p = np.c_[latent, predict_class,pre_class]
# att = np.c_[att, predict_class,pre_class]
# all_line = adata.shape[0]
# n_line = 0
# adata_list = []
#
#
# expdata = pd.DataFrame(todense(adata[n_line:n_line+min(n_step,(all_line-n_line))]),index=np.array(adata[n_line:n_line+min(n_step,(all_line-n_line))].obs_names).tolist(), columns=np.array(adata.var_names).tolist())
#
#
# with torch.no_grad():
#     for step, data in enumerate(data_loader):
#         exp = data
#         lat, pre, weights = model(exp.to(device))
#         pre = torch.squeeze(pre).cpu()
#         pre = F.softmax(pre,1)
#         predict_class = np.empty(shape=0)
#         pre_class = np.empty(shape=0)
#         for i in range(len(pre)):
#             if torch.max(pre, dim=1)[0][i] >= cutoff:
#                 predict_class = np.r_[predict_class,torch.max(pre,dim=1)[1][i].numpy()]
#             else:
#                 predict_class = np.r_[predict_class,n_c]
#             pre_class = np.r_[pre_class,torch.max(pre, dim=1)[0][i]]
#         l_p = torch.squeeze(lat).cpu().numpy()
#         att = torch.squeeze(weights).cpu().numpy()
#         meta = np.c_[predict_class,pre_class]
#         meta = pd.DataFrame(meta)
#         meta.columns = ['Prediction','Probability']
#         meta.index = meta.index.astype('str')
#         if laten:
#             l_p = l_p.astype('float32')
#             new = sc.AnnData(l_p, obs=meta)
#         else:
#             att = att[:, 0:(len(pathway) - n_unannotated)]
#             att = att.astype('float32')
#             varinfo = pd.DataFrame(pathway.iloc[0:len(pathway) - n_unannotated, 0].values,index=pathway.iloc[0:len(pathway) - n_unannotated, 0], columns=['pathway_index'])
#             new = sc.AnnData(att, obs=meta, var=varinfo)
#         adata_list.append(new)
#
#
#
# new = ad.concat(adata_list)
# new.obs.index = adata.obs.index
# new.obs['Prediction'] = new.obs['Prediction'].map(dic)
# new.obs[adata.obs.columns] = adata.obs[adata.obs.columns].values