import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from scipy.sparse import load_npz

train_chr=[1,4,5,6,7,9,10,11,13,14,15,16,17,18,19,20,22]
valid_chr=[3,12]
test_chr=[2,8,21]
k_adj=50
k_neigh=5

with open('neighbor_list/hic_list_%s_%s_3.pickle'%(k_adj,k_neigh), 'rb') as f:
    neighs = pickle.load(f)

class TrainDataset(Dataset):
    def __init__(self):
        nums=[]
        nums.append(0)
        for chr in train_chr:
            nums.append(neighs[chr].shape[0]+1)
        temp_dic={}
        for chr in train_chr:
            index=train_chr.index(chr)
            temp_dic[chr]=neighs[chr]+np.sum(nums[:index+1])
        self.x_idx=np.vstack([temp_dic[chr] for chr in train_chr])
        temp=[]
        for chr1 in train_chr:
            yl = load_npz('labels/ihec_labels/chr%s.npz' % chr1)
            temp.append(yl.toarray())
        temp_y=np.vstack([temp[i] for i in range(len(temp))])
        self.y = torch.FloatTensor(temp_y)
        self.num = self.y.shape[0]
    def __getitem__(self, index):
        return self.x_idx[index], self.y[index]
    def __len__(self):
        return self.num

class TestDataset(Dataset):
    def __init__(self):
        nums = []
        nums.append(0)
        for chr in test_chr:
            nums.append(neighs[chr].shape[0] + 1)
        temp_dic = {}
        for chr in test_chr:
            index = test_chr.index(chr)
            temp_dic[chr] = neighs[chr] + np.sum(nums[:index + 1])
        self.x_idx = np.vstack([temp_dic[chr] for chr in test_chr])
        temp = []
        for chr1 in test_chr:
            yl = load_npz('labels/ihec_labels/chr%s.npz' % chr1)
            temp.append(yl.toarray())
        temp_y = np.vstack([temp[i] for i in range(len(temp))])
        self.y = torch.FloatTensor(temp_y)
        self.num = self.y.shape[0]

    def __getitem__(self, index):
        return self.x_idx[index], self.y[index]
    def __len__(self):
        return self.num
class ValidDataset(Dataset):
    def __init__(self):
        nums = []
        nums.append(0)
        for chr in valid_chr:
            nums.append(neighs[chr].shape[0] + 1)
        temp_dic = {}
        for chr in valid_chr:
            index = valid_chr.index(chr)
            temp_dic[chr] = neighs[chr] + np.sum(nums[:index + 1])
        self.x_idx = np.vstack([temp_dic[chr] for chr in valid_chr])
        temp = []
        for chr1 in valid_chr:
            yl = load_npz('labels/ihec_labels/chr%s.npz' % chr1)
            temp.append(yl.toarray())
        temp_y = np.vstack([temp[i] for i in range(len(temp))])
        self.y = torch.FloatTensor(temp_y)
        self.num = self.y.shape[0]
    def __getitem__(self, index):
        return self.x_idx[index], self.y[index]
    def __len__(self):
        return self.num