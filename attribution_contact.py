import numpy as np
from torch.utils.data import DataLoader
import torch,time,argparse
from graph_layer import ECHO
import torch.optim as optim
import torch.nn as nn
import pickle
from scipy.sparse import csr_matrix,save_npz
from graph_dataset import TestDataset
parser = argparse.ArgumentParser()
parser.add_argument('--seq_length', type=int, default=1000, help='sequence length')
parser.add_argument('--lr', type=float, default=0.5, help='Learning rate.')
parser.add_argument('--pre_length', type=int, default=2600, help='pretrain model embed length')
parser.add_argument('--label_size', type=int, default=2583)
parser.add_argument('--k_adj', type=int, default=50,help='adjacent neighbors')
parser.add_argument('--k_neigh', type=int, default=10,help='sequence neighbors')
parser.add_argument('--pre_model', type=str, choices=['deepsea','expecto','danq'], default='expecto')
parser.add_argument('--cell_line', type=str,default='')
parser.add_argument('--chromatin_feature', type=str,default='CTCF')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
testloader=DataLoader(dataset=TestDataset(),batch_size=1,shuffle=False, num_workers=2)

test_chr=[2,8,21]
dir_path='/nfs/turbo/umms-drjieliu/usr/zzh/deepchrom/'
with open(dir_path+'hidden_feature/hidden_auc_%s_%s.pickle'%(args.pre_model,args.pre_length), 'rb') as f:
    hidden_feas=pickle.load(f)
from os import listdir
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
test_inputs = np.vstack([np.vstack((hidden_feas[chr],np.zeros((1,args.pre_length),dtype=np.float32)))
                          for chr in test_chr])
test_inputs=torch.tensor(test_inputs)
print('inputs load finished')
graph_model=ECHO(args.label_size,args.k_adj,args.k_neigh).to(device)
# graph_model.load_state_dict(torch.load('models/finetune/echo_auc_%s_%s_%s_%s.pt' %
#     (args.pre_model, args.pre_length, args.k_adj, args.k_neigh)))
graph_model.load_state_dict(torch.load('models/echo_auc_expecto_2600_50_10.pt'))
graph_model.eval()
loss_func = nn.BCEWithLogitsLoss()
attribution_adj=[]
attribution_neigh=[]
test_losses=[]
for step, (test_x_idx, test_batch_y) in enumerate(testloader):
    t = time.time()
    xidx = test_x_idx.flatten()
    xfea = test_inputs[xidx, :].to(device)
    test_batch_y = test_batch_y.to(device)
    xfea = xfea.reshape(test_batch_y.shape[0], args.k_adj + args.k_neigh + 1, args.pre_length)
    att1=torch.eye(args.k_adj,requires_grad=True)\
        .reshape(1,args.k_adj,args.k_adj).repeat(xfea.shape[0],1,1).float().to(device)
    temp_n=args.k_adj + args.k_neigh + 1
    att2 = torch.eye(args.k_neigh + 1, requires_grad=True)\
        .reshape(1, args.k_neigh + 1, args.k_neigh + 1).repeat(xfea.shape[0], 1, 1).float().to(device)
    xfea1 = xfea[:, :args.k_adj, :]
    xfea2 = xfea[:, args.k_adj:temp_n, :]
    xfea1=torch.matmul(att1,xfea1)
    xfea2=torch.matmul(att2,xfea2)
    cout = graph_model(xfea1, xfea2)
    # out=torch.sum(cout,1)
    out = torch.sum(cout[:,np.array(label_idx)], 1)
    att1.retain_grad()
    att2.retain_grad()
    out.backward()
    attribution_adj.append(att1.grad.data.cpu().detach().numpy())
    attribution_neigh.append(att2.grad.data.cpu().detach().numpy())
    if step % 20000 == 0:
        print(time.time() - t)
    if step>=hidden_feas[2].shape[0]:
        break
# np.save('model_validation/contact_attribution/test_adj_H3K4me3.npy',np.array(attribution_adj))
# np.save('model_validation/contact_attribution/test_neigh_H3K4me3.npy',np.array(attribution_neigh))
# test_adj=np.load('model_validation/contact_attribution/test_adj_H3K4me3.npy', allow_pickle=True)
# test_neigh=np.load('model_validation/contact_attribution/test_neigh_H3K4me3.npy', allow_pickle=True)

def get_diag(matrix):
    temp=[]
    for i in range(matrix.shape[0]):
        temp.append(np.diag(matrix[i,:,:]))
    return np.array(temp)
# test_adj=np.array(attribution_adj)
# test_neigh=np.array(attribution_neigh)
adjs_att=np.vstack([get_diag(attribution_adj[i]) for i in range(len(attribution_adj))])
neighs_att=np.vstack([get_diag(attribution_neigh[i]) for i in range(len(attribution_neigh))])
print(adjs_att.shape,neighs_att.shape)
with open(dir_path+'neighbor_list/neighbor_list_50_5.pickle','rb') as file:
    neigh_lists=pickle.load(file)

# get the attribution scores on Micro-C contacts in chr2
neighlist=neigh_lists[2]
num=neighlist.shape[0]
sample_adj_att=adjs_att[:num,:]
sample_neigh_att=neighs_att[:num,:]
neighlist_adj=neighlist[:,:args.k_adj]
neighlist_neigh=neighlist[:,args.k_adj:]
row=[]
data=[]
col=[]
for i in range(num):
    temp=[]
    temp_idx=[]
    for idx in range(neighlist_adj[i,:].shape[0]):
        if neighlist_adj[i,idx]!=num:
            temp.append(neighlist_adj[i,idx])
            temp_idx.append(idx)
    if temp:
        for t in range(len(temp)):
            row.append(i)
            col.append(temp[t])
            data.append(np.abs(sample_adj_att[i,temp_idx[t]]))
adj_attmatrix=csr_matrix((data, (row, col)), shape=(num, num))

row=[]
data=[]
col=[]
for i in range(num):
    temp=[]
    temp_idx=[]
    for idx in range(neighlist_neigh[i,:].shape[0]):
        if neighlist_neigh[i,idx]!=num:
            temp.append(neighlist_neigh[i,idx])
            temp_idx.append(idx)
    if temp:
        for t in range(len(temp)):
            row.append(i)
            col.append(temp[t])
            data.append(np.abs(sample_neigh_att[i,temp_idx[t]]))
neigh_attmatrix=csr_matrix((data, (row, col)), shape=(num, num))
def maximum (A, B):
    BisBigger = A-B
    BisBigger.data = np.where(BisBigger.data < 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)
attribution_matrix=maximum(adj_attmatrix,neigh_attmatrix)
if args.cell_line:
    save_npz('attribution_matrix_%s_%s.npz'%(args.cell_line,args.chromatin_feature),attribution_matrix)
else:
    save_npz('attribution_matrix_%s.npz' % args.chromatin_feature,attribution_matrix)