import re,pickle,os,argparse
import torch
import numpy as np
from graph_layer import ECHO
import torch.optim as optim
import torch.nn as nn
import pickle
import multiprocessing as mp
from sklearn import metrics
from torch.utils.data import DataLoader
import time
from graph_dataset import TrainDataset,ValidDataset,TestDataset
parser = argparse.ArgumentParser()
parser.add_argument('--seq_length', type=int, default=1000, help='sequence length')
parser.add_argument('--lr', type=float, default=0.5, help='Learning rate.')
parser.add_argument('--pre_length', type=int, default=2600, help='pretrain model embed length')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--label_size', type=int, default=2583)
parser.add_argument('--epochs', type=int, default=64)
parser.add_argument('--k_adj', type=int, default=50,help='number of spatial neighbors')
parser.add_argument('--k_neigh', type=int, default=10,help='number of sequential neighbors')
parser.add_argument('--pre_model', type=str, choices=['deepsea','expecto','danq'], default='expecto')
parser.add_argument('--metric', type=str, choices=['loss','auc'], default='loss')
parser.add_argument('--checkpoint', default=False, action='store_true')
parser.add_argument('--test', default=False, action='store_true')
args = parser.parse_args()
cell_lines=['GM12878','H1-hESC','IMR90','K562']
test_chrs=[2,8,21]
# positive_compart = {}
# negative_compart = {}
dir_path='/nfs/turbo/umms-drjieliu/usr/zzh/deepchrom/'
with open(dir_path+'input_sample_poi.pickle','rb') as file:
    sample_locs=pickle.load(file)
def search_loc(chr,point,start,end):
    find_locs=[]
    while sample_locs[chr][point]<end and point<=sample_locs[chr].shape[0]:
        if sample_locs[chr][point]>= start:
            find_locs.append(point)
        point+=1
    return find_locs,point
def find_compartment(file):
    positive_compart = {}
    negative_compart = {}
    for chr in test_chrs:
        positive_compart[chr] = []
        negative_compart[chr] = []
    visited_chr=set()
    with open(file,'r') as f:
        contents=f.readlines()
        for line in contents:
            compart=line.strip().split('\t')
            try:
                chr=int(re.findall('\d+',compart[0])[0])
                if chr in test_chrs and compart[-1]!='nan':
                    if chr not in visited_chr:
                        point=0
                        visited_chr.add(chr)
                    start=int(compart[1])
                    end=int(compart[2])
                    find_locs,point=search_loc(chr,point,start,end)
                    compart_value=float(compart[-1])
                    if compart_value>0:
                        positive_compart[chr].extend(find_locs)
                    else:
                        negative_compart[chr].extend(find_locs)
            except Exception:
                pass
    return positive_compart,negative_compart

def cal_auc(v):
    true, pred = v
    idx = np.where(true > 0)
    if idx[0].tolist():
        auc=metrics.roc_auc_score(true,pred)
        ap=metrics.average_precision_score(true,pred)
    else:
        auc=np.nan
        ap=np.nan
    return auc,ap
def best_param(preds,targets,preserve,name):

    print(preds.shape)
    pool = mp.Pool(32)
    result = pool.map_async(cal_auc, ((targets[:, i], preds[:, i]) for i in range(preds.shape[1])))
    pool.close()
    pool.join()
    r = np.array(result.get())
    r=r[~np.isnan(r)].reshape(-1,2)
    print(r.shape)
    if preserve:
        np.save('positive_%s.npy'%name,r)
    else:
        np.save('negative_%s.npy' % name, r)
    return np.mean(r[:,0]),np.mean(r[:,1])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
with open(dir_path+'hidden_feature/hidden_auc_%s_%s.pickle'%(args.pre_model,args.pre_length), 'rb') as f:
    hidden_feas=pickle.load(f)
print('inputs load finished')
graph_model = ECHO(args.label_size, args.k_adj, args.k_neigh)
graph_model.to(device)
optimizer = optim.SGD(graph_model.parameters(), lr=args.lr,momentum=0.9,weight_decay=1e-6)
loss_func = nn.BCEWithLogitsLoss()

print('model test')
# graph_model.load_state_dict(torch.load(dir_path+'models/finetune/echo_loss_%s_%s_%s_%s.pt' %
#                            (args.pre_model, args.pre_length, args.k_adj, args.k_neigh)))
graph_model.load_state_dict(torch.load(dir_path+'models/finetune/graph_auc_expecto_2600_50_10_0_v1.pt'))

test_inputs = np.vstack([np.vstack((hidden_feas[chr], np.zeros((1, args.pre_length), dtype=np.float32)))
                             for chr in test_chrs])
test_inputs = torch.tensor(test_inputs)
testloader = DataLoader(dataset=TestDataset(), batch_size=args.batchsize, shuffle=False, num_workers=2)
pred_eval = []
target_eval = []
test_losses=[]
graph_model.eval()
for step, (test_x_idx, test_batch_y) in enumerate(testloader):
    t = time.time()
    xidx = test_x_idx.flatten()
    xfea = test_inputs[xidx, :].to(device)
    test_batch_y = test_batch_y.to(device)
    xfea = xfea.reshape(test_batch_y.shape[0], args.k_adj + args.k_neigh + 1, args.pre_length)
    xfea1 = xfea[:, :args.k_adj, :]
    xfea2 = xfea[:, args.k_adj:args.k_adj + args.k_neigh + 1, :]
    out = graph_model(xfea1, xfea2)
    loss = loss_func(out, test_batch_y)
    cur_loss = loss.item()
    test_losses.append(cur_loss)
    pred_eval.append(torch.sigmoid(out).cpu().data.detach().numpy())
    target_eval.append(test_batch_y.cpu().data.detach().numpy())
    if step % 1000 == 0:
        print( "step:", '%04d' % (step + 1), "loss=",
                  "{:.7f}".format(cur_loss),
                  "time=", "{:.5f}".format(time.time() - t)
                  )
preds = np.vstack([pred_eval[i] for i in range(len(pred_eval))])
targets= np.vstack([target_eval[i] for i in range(len(target_eval))])
print(preds.shape,targets.shape)
test_loss = np.average(test_losses)
print('test Loss is %s' % test_loss)
    # auc_score, ap_score=best_param(pred_eval, target_eval,1)
    # print('mean auc is %s, mean ap is %s' % (auc_score, ap_score))
compart_path='/nfs/turbo/umms-drjieliu/proj/4dn/data/bulkHiC/HiC_compartments/'
cell_name=['GM12878_','H1_','IMR-90_','K562_']

path='/nfs/turbo/umms-drjieliu/proj/4dn/data/Epigenomic_Data/chromatin_feature_hg38/tf'
files_dir= [f for f in os.listdir(path)]
path='/nfs/turbo/umms-drjieliu/proj/4dn/data/Epigenomic_Data/chromatin_feature_hg38/ihec_histone'
files_hm= [f for f in os.listdir(path)]
path='/nfs/turbo/umms-drjieliu/proj/4dn/data/Epigenomic_Data/chromatin_feature_hg38/dnase'
files_dnase= [f for f in os.listdir(path)]
files_dir.extend(files_hm)
files_dir.extend(files_dnase)
files_dir=np.array(files_dir)
for i in range(len(cell_lines)):
    label_idx=[]
    for idx in range(files_dir.shape[0]):
        if cell_name[i] in files_dir[idx]:
            label_idx.append(idx)
    print(files_dir[np.array(label_idx)])
    files=os.listdir(compart_path+cell_lines[i])
    for f in files:
        if '.bedGraph' in f:
            file_path=compart_path+cell_lines[i]+'/'+f
    positive_compart,negative_compart=find_compartment(file_path)
    temp_index = []
    for chr in test_chrs:
        num = hidden_feas[2].shape[0]
        num1 = hidden_feas[8].shape[0]
        temp_index.append(np.array(positive_compart[2]))
        temp_index.append(np.array(positive_compart[8]) + num)
        temp_index.append(np.array(positive_compart[21]) + num + num1)
    positive_loc_idx = np.concatenate([temp_index[i] for i in range(len(temp_index))])
    print(positive_loc_idx.shape)
    print(positive_loc_idx)
    temp_index = []
    for chr in test_chrs:
        num = hidden_feas[2].shape[0]
        num1 = hidden_feas[8].shape[0]
        temp_index.append(np.array(negative_compart[2]))
        temp_index.append(np.array(negative_compart[8]) + num)
        temp_index.append(np.array(negative_compart[21]) + num + num1)
    negative_loc_idx = np.concatenate([temp_index[i] for i in range(len(temp_index))])
    print(negative_loc_idx.shape)
    print(negative_loc_idx)

    auc_score, ap_score = best_param(preds[positive_loc_idx,:][:,np.array(label_idx)],targets[positive_loc_idx,:][:,np.array(label_idx)], 1,cell_lines[i])
    print(auc_score,ap_score)
    auc_score, ap_score = best_param(preds[negative_loc_idx, :][:, np.array(label_idx)],targets[negative_loc_idx,:][:,np.array(label_idx)], 0, cell_lines[i])
    print(auc_score, ap_score)
