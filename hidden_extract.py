import torch,argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pretrain_layer import Expecto,DanQ,DeepSEA
import pickle
device = torch.device( 'cuda:0' if torch.cuda.is_available() else'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--seq_length', type=int, default=1000, help='sequence length')
parser.add_argument('--length', type=int, default=2600, help='length of hidden representation')
parser.add_argument('--pre_model', type=str, choices=['deepsea','expecto','danq']
                    , default='expecto',help='pre-train models')
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--label_size', type=int, default=2583)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()
if args.pre_model=='deepsea':
    model=DeepSEA(args.label_size,args.seq_length,args.length)
elif args.pre_model=='expecto':
    model=Expecto(args.label_size,args.seq_length,args.length)
else:
    model=DanQ(args.label_size,args.seq_length,args.length)
model.to(device)
model.load_state_dict(torch.load('models/%s_auc_%s.pt'%(args.pre_model,args.length)))
hidden={}
for chr in range(1,23):
    print(chr)
    features=np.load('inputs/chr%s.npy'%chr)
    model.eval()
    fea = torch.tensor(features).float()
    h_fea = np.zeros((fea.shape[0],args.length),dtype="float32")
    cur_idx = 0
    for idx in range(fea.shape[0] // 200):
        batch = fea[cur_idx:cur_idx + 200, :, :].to(device)
        _, xfea = model(batch)
        temp_fea = xfea.cpu().data.detach().numpy()
        h_fea[cur_idx:cur_idx + 200, :] = np.float32(temp_fea)
        cur_idx += 200
        if idx % 100 == 0:
            print('%s finished' % idx)
    if fea.shape[0] % 200:
        batch = fea[cur_idx:, :, :].to(device)
        _, xfea = model(batch)
        temp_fea = xfea.cpu().data.detach().numpy()
        h_fea[cur_idx:, :] = np.float32(temp_fea)
    hidden[chr] = h_fea
with open('hidden_feature/hidden_auc_%s_%s.pickle'%(args.pre_model,args.length),'wb') as f:
    pickle.dump(hidden,f)