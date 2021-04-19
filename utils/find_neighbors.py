# this code is to sample and order neighbors by contact values to generate neighbor sets.

import pickle,argparse
import numpy as np
from scipy.sparse import csr_matrix
parser = argparse.ArgumentParser()
parser.add_argument('--k_adj', type=int, default=50,help='adjacent neighbors')
parser.add_argument('--k_neigh', type=int, default=5,help='sequence neighbors')
args = parser.parse_args()

with open('adj_matrix/top_adjacency_matrix_2.pickle','rb') as f:
    adjs=pickle.load(f)

def find_adjs(array):
    nums=array.nnz
    idxs=np.argsort(array.data)[::-1]
    sort_idx=array.indices[idxs]
    if nums>=args.k_adj:
        idx=sort_idx[:args.k_adj]
        temp=0
    else:
        idx=sort_idx[:nums]
        temp=args.k_adj-nums
    return idx.tolist(),temp
neighs={}
for chr in range(1,23):
    print(chr)
    adj_matrix=csr_matrix(adjs[chr])
    len_adj=adj_matrix.shape[0]
    print(len_adj)
    neigh_list=[len_adj]*args.k_neigh+[n for n in range(len_adj)]+[len_adj]*args.k_neigh
    temp_list=[]
    for i in range(len_adj):
        idx,adds=find_adjs(adj_matrix[i,:])
        temp=idx+adds*[len_adj]
        temp.extend(neigh_list[i:i+2*args.k_neigh+1])
        temp_list.append(temp)
    neighs[chr]=np.array(temp_list)
with open('neighbor_list/neighbor_list_%s_%s.pickle' % (args.k_adj,args.k_neigh), 'wb') as f:
    pickle.dump(neighs,f)