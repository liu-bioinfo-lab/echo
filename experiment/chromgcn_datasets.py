import torch
import numpy as np
from scipy.sparse import coo_matrix
from scipy import sparse
from torch.utils.data import Dataset, DataLoader

import pickle

pwd = '/nfs/turbo/umms-drjieliu/usr/zzh/chromGCN_data/GM12878/1000/'
sets=['train','test','valid']
inputs=[]
inputs_rev=[]
labels=[]
adjs=[]
for ss in sets:
    with open(pwd + 'processed_data/%s_hiddens.pickle'%ss, 'rb') as f:
        tempi= pickle.load(f)
    with open(pwd + 'processed_data/%s_hiddens_revcom.pickle'%ss, 'rb') as f:
        tempi_rev= pickle.load(f)
    with open(pwd + 'processed_data/%s_labels.pickle'%ss, 'rb') as f:
        templ = pickle.load(f)
    with open(pwd+'hic/%s_graphs_500000_SQRTVCnorm.pkl'%ss,'rb') as f:
        tempa=pickle.load(f)
    inputs.append(tempi)
    inputs_rev.append(tempi_rev)
    labels.append(templ)
    adjs.append(tempa)
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)).astype(float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def create_constant_graph(constant_range,x_size):
    diagonals,indices = [],[]
    for i in range(-constant_range,constant_range+1):
        if i != 0:
            diagonals.append(np.ones(x_size-abs(i)))
            indices.append(i)
    split_adj = sparse.diags(diagonals, indices).tocoo()
    return split_adj
def adj_gen(adj):
    x_size=adj.shape[0]
    const_adj = create_constant_graph(7, x_size)
    coo=coo_matrix(adj)
    coo=coo+const_adj+sparse.eye(adj.shape[0])
    coo=normalize(coo)
    coo=coo.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))
class train_sets():
    def __init__(self):
        self.input={}
        self.input_rev={}
        self.label={}
        self.adj={}
    def get_data(self):
        for k in inputs[0].keys():
            self.input[k]=torch.FloatTensor(inputs[0][k])
            self.input_rev[k]=torch.FloatTensor(inputs_rev[0][k])
            self.label[k] = torch.FloatTensor(labels[0][k])
            self.adj[k]=adj_gen(adjs[0]['chr%s'%k])
        return self.input,self.input_rev,self.label,self.adj
class test_sets():
    def __init__(self):
        self.input={}
        self.input_rev={}
        self.label={}
        self.adj={}
    def get_data(self):
        for k in inputs[1].keys():
            self.input[k]=torch.FloatTensor(inputs[1][k])
            self.input_rev[k] = torch.FloatTensor(inputs_rev[1][k])
            self.label[k] = torch.FloatTensor(labels[1][k])
            self.adj[k]=adj_gen(adjs[1]['chr%s'%k])
        return self.input,self.input_rev,self.label,self.adj
class valid_sets():
    def __init__(self):
        self.input={}
        self.input_rev={}
        self.label={}
        self.adj={}
    def get_data(self):
        for k in inputs[2].keys():
            self.input[k]=torch.FloatTensor(inputs[2][k])
            self.input_rev[k]=torch.FloatTensor(inputs_rev[2][k])
            self.label[k] = torch.FloatTensor(labels[2][k])
            self.adj[k]=adj_gen(adjs[2]['chr%s'%k])
        return self.input,self.input_rev,self.label,self.adj