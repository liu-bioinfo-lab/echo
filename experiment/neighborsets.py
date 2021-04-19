import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

device = torch.device( 'cuda:0' if torch.cuda.is_available() else'cpu')
pwd = '/nfs/turbo/umms-drjieliu/usr/zzh/chromGCN_data/GM12878/1000/'
train_inputs=[]
train_rev_inputs=[]
train_labels=[]
train_adjs=[]
with open(pwd + 'processed_data/train_hiddens.pickle', 'rb') as f:
    trains= pickle.load(f)
with open(pwd + 'processed_data/train_hiddens_revcom.pickle', 'rb') as f:
    trains_rev= pickle.load(f)
with open(pwd + 'processed_data/train_labels.pickle', 'rb') as f:
    train_l = pickle.load(f)
with open(pwd+'processed_data/train_adjs_k=30.pickle','rb') as f:
    train_m=pickle.load(f)
for i in trains.keys():
    temp_x=torch.FloatTensor(trains[i])
    temp_x_rev=torch.FloatTensor(trains_rev[i])
    temp_y=torch.FloatTensor(train_l[i])
    temp_a=torch.FloatTensor(train_m[i])
    assert temp_x.shape[0]==temp_y.shape[0]
    train_inputs.append(temp_x)
    train_rev_inputs.append(temp_x_rev)
    train_labels.append(temp_y)
    train_adjs.append(temp_a)
valid_inputs=[]
valid_rev_inputs=[]
valid_labels=[]
valid_adjs=[]
with open(pwd + 'processed_data/valid_hiddens.pickle', 'rb') as f:
    valids= pickle.load(f)
with open(pwd + 'processed_data/valid_hiddens_revcom.pickle', 'rb') as f:
    valids_rev= pickle.load(f)
with open(pwd + 'processed_data/valid_labels.pickle', 'rb') as f:
    valid_l = pickle.load(f)
with open(pwd+'processed_data/valid_adjs_k=30.pickle','rb') as f:
    valid_m=pickle.load(f)
for i in valids.keys():
    temp_x=torch.FloatTensor(valids[i])
    temp_x_rev = torch.FloatTensor(valids_rev[i])
    temp_y=torch.FloatTensor(valid_l[i])
    temp_a = torch.FloatTensor(valid_m[i])
    assert temp_x.shape[0]==temp_y.shape[0]
    valid_inputs.append(temp_x)
    valid_rev_inputs.append(temp_x_rev)
    valid_labels.append(temp_y)
    valid_adjs.append(temp_a)
test_inputs=[]
test_rev_inputs=[]
test_labels=[]
test_adjs=[]
with open(pwd + 'processed_data/test_hiddens.pickle', 'rb') as f:
    tests= pickle.load(f)
with open(pwd + 'processed_data/test_hiddens_revcom.pickle', 'rb') as f:
    tests_rev= pickle.load(f)
with open(pwd + 'processed_data/test_labels.pickle', 'rb') as f:
    test_l = pickle.load(f)
with open(pwd+'processed_data/test_adjs_k=30.pickle','rb') as f:
    test_m=pickle.load(f)
for i in tests.keys():
    temp_x=torch.FloatTensor(tests[i])
    temp_x_rev = torch.FloatTensor(tests_rev[i])
    temp_y=torch.FloatTensor(test_l[i])
    temp_a = torch.FloatTensor(test_m[i])
    assert temp_x.shape[0]==temp_y.shape[0]
    test_inputs.append(temp_x)
    test_rev_inputs.append(temp_x_rev)
    test_labels.append(temp_y)
    test_adjs.append(temp_a)

def generate_fea(fea, inx):
    num = fea.shape[0]
    zeros = torch.zeros((1, fea.shape[1]), dtype=float)
    fea = torch.cat((fea, zeros))
    inx = inx.flatten()
    v = torch.ones_like(inx, dtype=float)
    poi = torch.FloatTensor(np.arange(inx.shape[0]))
    i = torch.cat((poi.reshape(1, -1), inx.reshape(1, -1))).long()
    w = torch.sparse.FloatTensor(i, v, torch.Size([inx.shape[0], fea.shape[0]]))
    # self.w.requires_grad = True
    temp_fea = torch.sparse.mm(w, fea)
    new_fea = temp_fea.reshape(num, -1, fea.shape[1]).float()
    return new_fea

def neighbor(fea,lens):
    nums=fea.shape[0]
    zerop= torch.zeros((lens,fea.shape[1]),dtype=float)
    temp_fea=torch.cat((zerop,fea,zerop),0)
    new_fea=torch.cat([temp_fea[i-lens:i+lens+1,:] for i in range(lens,lens+nums)])
    new_fea=new_fea.reshape(nums,2*lens+1,-1).float()
    # print(new_fea.shape)
    return new_fea



class TrainDataset(Dataset):
    def __init__(self):
        feas=[]
        feas_rev=[]
        labels=[]
        neigh=[]
        neigh_rev=[]
        for i in range(len(train_inputs)):
            labels.append(torch.FloatTensor(train_labels[i]))
            temp_x=torch.FloatTensor(train_inputs[i])
            temp_x_rev=torch.FloatTensor(train_rev_inputs[i])
            neigh.append(neighbor(temp_x,5))
            feas.append(generate_fea(temp_x,train_adjs[i]))

            neigh_rev.append(neighbor(temp_x_rev, 5))
            feas_rev.append(generate_fea(temp_x_rev, train_adjs[i]))
        self.x=torch.cat([feas[i] for i in range(len(feas))])
        self.x1 = torch.cat([neigh[i] for i in range(len(neigh))])

        self.x_rev=torch.cat([feas_rev[i] for i in range(len(feas))])
        self.x1_rev = torch.cat([neigh_rev[i] for i in range(len(neigh))])
        self.y=torch.cat([labels[i] for i in range(len(labels))])
        self.num= self.x1.shape[0]
        print(self.x.shape)
        print(self.y.shape)
        print(self.x_rev.shape)
    def __getitem__(self, index):
        return self.x[index],self.x1[index],self.x_rev[index],self.x1_rev[index], self.y[index]
    def __len__(self):
        return self.num
class ValiDataset(Dataset):
    def __init__(self):
        feas = []
        feas_rev = []
        labels = []
        neigh = []
        neigh_rev = []
        for i in range(len(valid_inputs)):
            labels.append(torch.FloatTensor(valid_labels[i]))
            temp_x = torch.FloatTensor(valid_inputs[i])
            temp_x_rev = torch.FloatTensor(valid_rev_inputs[i])
            neigh.append(neighbor(temp_x, 5))
            feas.append(generate_fea(temp_x, valid_adjs[i]))

            neigh_rev.append(neighbor(temp_x_rev, 5))
            feas_rev.append(generate_fea(temp_x_rev, valid_adjs[i]))
        self.x = torch.cat([feas[i] for i in range(len(feas))])
        self.x1 = torch.cat([neigh[i] for i in range(len(neigh))])

        self.x_rev = torch.cat([feas_rev[i] for i in range(len(feas))])
        self.x1_rev = torch.cat([neigh_rev[i] for i in range(len(neigh))])
        self.y = torch.cat([labels[i] for i in range(len(labels))])
        self.num = self.x1.shape[0]
        print(self.x.shape)
        print(self.y.shape)
        print(self.x_rev.shape)
    def __getitem__(self, index):
        return self.x[index], self.x1[index], self.x_rev[index], self.x1_rev[index], self.y[index]
    def __len__(self):
        return self.num

class TestDataset(Dataset):
    def __init__(self):
        feas = []
        feas_rev = []
        labels = []
        neigh = []
        neigh_rev = []
        for i in range(len(test_inputs)):
            labels.append(torch.FloatTensor(test_labels[i]))
            temp_x = torch.FloatTensor(test_inputs[i])
            temp_x_rev = torch.FloatTensor(test_rev_inputs[i])
            neigh.append(neighbor(temp_x, 5))
            feas.append(generate_fea(temp_x, test_adjs[i]))

            neigh_rev.append(neighbor(temp_x_rev, 5))
            feas_rev.append(generate_fea(temp_x_rev, test_adjs[i]))
        self.x = torch.cat([feas[i] for i in range(len(feas))])
        self.x1 = torch.cat([neigh[i] for i in range(len(neigh))])

        self.x_rev = torch.cat([feas_rev[i] for i in range(len(feas))])
        self.x1_rev = torch.cat([neigh_rev[i] for i in range(len(neigh))])
        self.y = torch.cat([labels[i] for i in range(len(labels))])
        self.num = self.x1.shape[0]
        print(self.x.shape)
        print(self.y.shape)
        print(self.x_rev.shape)
    def __getitem__(self, index):
        return self.x[index], self.x1[index], self.x_rev[index], self.x1_rev[index], self.y[index]
    def __len__(self):
        return self.num