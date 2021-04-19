import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from scipy.sparse import load_npz
train_chr=[1,4,5,6,7,9,10,11,13,14,15,16,17,18,19,20,22]
valid_chr=[3,12]
test_chr=[2,8,21]
dir_path='/nfs/turbo/umms-drjieliu/usr/zzh/deepchrom/'
class TrainDataset(Dataset):
    def __init__(self):
        temp=[]
        temp1 = []
        for chr in train_chr:
            xinputs = np.load(dir_path+'inputs/chr%s.npy' % chr)
            temp.append(xinputs)
            yl = load_npz(dir_path+'labels/ihec_labels/chr%s.npz' % chr)
            temp1.append(yl.toarray())
        train_inputs = np.vstack([temp[i] for i in range(len(temp))])
        self.x = torch.tensor(train_inputs)
        temp_y=np.vstack([temp1[i] for i in range(len(temp1))])
        self.y = torch.tensor(temp_y).float()
        self.num = self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.num

class TestDataset(Dataset):
    def __init__(self):
        temp = []
        for chr in test_chr:
            xinputs = np.load(dir_path+'inputs/chr%s.npy' % chr)
            temp.append(xinputs)
        test_inputs = np.vstack([temp[i] for i in range(len(temp))])
        self.x = torch.tensor(test_inputs)
        temp=[]
        for chr1 in test_chr:
            yl = load_npz(dir_path+'labels/ihec_labels/chr%s.npz' % chr1)
            temp.append(yl.toarray())
        temp_y=np.vstack([temp[i] for i in range(len(temp))])
        self.y = torch.FloatTensor(temp_y)
        self.num = self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.num
class ValidDataset(Dataset):
    def __init__(self):
        temp = []
        for chr in valid_chr:
            xinputs = np.load(dir_path+'inputs/chr%s.npy' % chr)
            temp.append(xinputs)
        valid_inputs = np.vstack([temp[i] for i in range(len(temp))])
        self.x = torch.tensor(valid_inputs)
        temp=[]
        for chr1 in valid_chr:
            yl = load_npz(dir_path+'labels/ihec_labels/chr%s.npz' % chr1)
            temp.append(yl.toarray())
        temp_y=np.vstack([temp[i] for i in range(len(temp))])
        self.y = torch.FloatTensor(temp_y)
        self.num = self.y.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.num