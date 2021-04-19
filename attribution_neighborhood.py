from scipy import sparse
import torch,pickle,argparse
import numpy as np
import torch.nn as nn
from pretrain_layer import Expecto,DanQ,DeepSEA
from graph_layer import ECHO
from scipy.sparse import load_npz,csr_matrix
from sklearn.preprocessing import normalize
from os import listdir
parser = argparse.ArgumentParser()
parser.add_argument('--seq_length', type=int, default=1000, help='sequence length')
parser.add_argument('--pre_length', type=int, default=2600, help='pretrain model embed length')
parser.add_argument('--label_size', type=int, default=2583)
parser.add_argument('--k_adj', type=int, default=50,help='adjacent neighbors')
parser.add_argument('--k_neigh', type=int, default=10,help='sequence neighbors')
parser.add_argument('--pre_model', type=str, choices=['deepsea','expecto','danq'], default='expecto')
parser.add_argument('--cell_line', type=str,default='')
parser.add_argument('--chromatin_feature', type=str,default='h3k4me3')
args = parser.parse_args()
dir_path='/nfs/turbo/umms-drjieliu/usr/zzh/deepchrom/'
if args.pre_model=='deepsea':
    model=DeepSEA(args.label_size,args.seq_length,args.pre_length)
elif args.pre_model=='expecto':
    model=Expecto(args.label_size,args.seq_length,args.pre_length)
else:
    model=DanQ(args.label_size,args.seq_length,args.pre_length)
device = torch.device( 'cuda:0' if torch.cuda.is_available() else'cpu')
model.to(device)
model.load_state_dict(torch.load(dir_path+'models/%s_auc_%s_v1.pt'%(args.pre_model,args.pre_length)))
graph_model=ECHO(args.label_size,args.k_adj,args.k_neigh)
graph_model.to(device)
graph_model.load_state_dict(torch.load(dir_path+'models/finetune/graph_auc_expecto_2600_50_10_0_v1.pt'))
# graph_model.load_state_dict(torch.load(
#      dir_path+'models/finetune/graph_auc_%s_%s_%s_%s_%s_%s.pt' %
#                            (args.pre_model, args.pre_length, args.k_adj, args.k_neigh, args.k_second, args.version)))
path='/nfs/turbo/umms-drjieliu/proj/4dn/data/Epigenomic_Data/chromatin_feature_hg38/tf'
files_dir= [f for f in listdir(path)]
path='/nfs/turbo/umms-drjieliu/proj/4dn/data/Epigenomic_Data/chromatin_feature_hg38/ihec_histone'
files_hm= [f for f in listdir(path)]
files_dir.extend(files_hm)

label_idx=[]
for i in range(len(files_dir)):
# cell-type specific
    if args.cell_line:
        if args.cell_line.upper() in files_dir[i].upper() and args.chromatin_feature.upper() in files_dir[i].upper():
            label_idx.append(i)
# general chromatin features in all cell lines
    else:
        if args.chromatin_feature.upper() in files_dir[i].upper():
            label_idx.append(i)
if label_idx:
    print('%s chromatin features found'%len(label_idx))
else:
    print('no chromatin feature is found')
label_idx=np.array(label_idx)
inspect_chr=2
labels=sparse.load_npz(dir_path+'labels/ihec_labels/chr%s.npz'%inspect_chr).toarray()
temp_label=np.sum(labels[:,label_idx],1)
label_locs=np.where(temp_label>0)[0]
print(label_locs.shape)

#load the corresponding attributed contact matrix
if args.cell_line:
    contact_matrix=load_npz('attribution_matrix_%s_%s.npz'%(args.cell_line,args.chromatin_feature))
else:
    contact_matrix=load_npz('attribution_matrix_%s.npz' % args.chromatin_feature)

contact_matrix = normalize(np.absolute(contact_matrix), norm='max', axis=1)
contact_matrix.setdiag(0)

# select contacts with high attribution scores
m = contact_matrix.multiply(contact_matrix >= 0.7)
m=csr_matrix(m[label_locs,:])
col=m.tocoo().col
row=m.tocoo().row
with open(dir_path+'input_sample_poi.pickle','rb') as file:
    sample_loc=pickle.load(file)
interactions=[]
for i in range(col.shape[0]):
    bin1=sample_loc[2][label_locs[row[i]]]
    bin2=sample_loc[2][col[i]]
# we add flank region to both upstream and downstream of each sequence, this ensures
# bin1 and bin2 have non-overlapping regions
    if np.abs(bin1-bin2)>800:
        interactions.append([label_locs[row[i]],col[i]])
print(len(interactions))


with open(dir_path+'neighbor_list/neighbor_list_50_5.pickle','rb') as f:
    adjs=pickle.load(f)
adj_matrix=adjs[inspect_chr]
inputs=np.load(dir_path+'inputs/chr%s.npy'%inspect_chr)
inputs=np.vstack((inputs,np.zeros((1,4,args.seq_length),dtype=np.int8)))

# colltect the gradient
gradients=[]
input_seqs=[]
print('training')
for inters in interactions:
    idx=inters[0]
    neigh=inters[1]
    idx_neigh=np.where(adj_matrix[idx,:]==neigh)[0]
    assert np.where(adj_matrix[idx,:]==idx)[0] ==55
    model.eval()
    graph_model.eval()
# identify the binding events of the chromatin features from multiple cell lines
    feature_idx = np.where(labels[idx,label_idx]>0)[0]
    feature_idx=label_idx[feature_idx]
    input_fea=torch.tensor(inputs[adj_matrix[idx,:],:,:]).squeeze().float().to(device)
    input_fea.requires_grad = True
    _, xfea = model(input_fea)
    xfea1 = xfea[:args.k_adj, :].unsqueeze(0)
    xfea2=xfea[args.k_adj:,:].unsqueeze(0)
    out = graph_model(xfea1, xfea2)
    feature_out=torch.mean(torch.sigmoid(out[:,feature_idx]),1)
# make sure that the binding events are successfully predicted with high prediction scores
    if feature_out>0.6:
        out = torch.sum(out[:, feature_idx], 1)
        out.backward()
        grads =  input_fea.grad.data.cpu().detach().numpy()
        # grad_input = (input_fea * input_fea.grad.data)[:, :, :].cpu().detach().numpy()
        gradients.append(grads[idx_neigh[0].item(),:,:])
        input_seqs.append(inputs[neigh,:,:])
    if len(gradients) >=400:
        print('stopping')
        break
if args.cell_line:
    np.save('%s_%s_grad.npy'%(args.cell_line,args.chromatin_feature),gradients)
    np.save('%s_%s_input.npy'%(args.cell_line,args.chromatin_feature),input_seqs)
else:
    np.save('%s_grad.npy' %args.chromatin_feature, gradients)
    np.save('%s_input.npy' %args.chromatin_feature, input_seqs)