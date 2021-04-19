import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import deepcnn1
import pickle
pwd = '/nfs/turbo/umms-drjieliu/usr/zzh/chromGCN_data/GM12878/1000/'
device = torch.device( 'cuda:0' if torch.cuda.is_available() else'cpu')
model=deepcnn1.Expecto(103,2000).to(device)
# model.load_state_dict(torch.load(pwd+'code/danqauc_embed.pt'))
model.load_state_dict(torch.load(pwd+'code/deepcnn_rev_nd.pt'))
for sets in ['train','valid','test']:
    with open(pwd + 'processed_data/%s_inputs.pickle'%sets, 'rb') as f:
        features = pickle.load(f)
    with open(pwd + 'processed_data/%s_inputs_revcom.pickle'%sets, 'rb') as handle:
        revcom_fea = pickle.load(handle)
    print('features load')
    model.eval()
    hidden={}
    hidden_rev={}
    for i in features.keys():
        fea=torch.tensor(features[i]).float()
        fea_rev=torch.tensor(revcom_fea[i]).float()
        h_fea=np.zeros((fea.shape[0],128))
        h_fea_rev=np.zeros((fea.shape[0],128))
        cur_idx=0
        for idx in range(fea.shape[0]//100):
            batch=fea[cur_idx:cur_idx+100,:,:].to(device)
            batch_rev=fea_rev[cur_idx:cur_idx+100,:,:].to(device)
            _,xfea=model(batch)
            _,xfea_rev=model(batch_rev)
            temp_fea=xfea.cpu().data.detach().numpy()
            temp_fea_rev=xfea_rev.cpu().data.detach().numpy()
            h_fea[cur_idx:cur_idx+100,:]=temp_fea
            h_fea_rev[cur_idx:cur_idx+100,:]=temp_fea_rev
            cur_idx+=100
            if idx %20==0:
                print('%s finished'%idx)
        if fea.shape[0] %100:
            batch = fea[cur_idx:,:, :].to(device)
            batch_rev=fea_rev[cur_idx:,:, :].to(device)
            _, xfea = model(batch)
            _, xfea_rev = model(batch_rev)
            temp_fea = xfea.cpu().data.detach().numpy()
            temp_fea_rev = xfea_rev.cpu().data.detach().numpy()
            h_fea[cur_idx:,:]=temp_fea
            h_fea_rev[cur_idx:,:]=temp_fea_rev
        hidden[i]=h_fea
        hidden_rev[i]=h_fea_rev
    with open(pwd + 'processed_data/%s_hiddens.pickle'%sets, 'wb') as f:
        pickle.dump(hidden,f)
    with open(pwd + 'processed_data/%s_hiddens_revcom.pickle'%sets, 'wb') as f:
        pickle.dump(hidden_rev,f)