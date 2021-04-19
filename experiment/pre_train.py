import numpy as np
import pickle
from sklearn import metrics
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
import time,argparse
from pre_layer import Expecto
device = torch.device( 'cuda:0' if torch.cuda.is_available() else'cpu')
print(device)
parser = argparse.ArgumentParser()
parser.add_argument('--test', default=False, action='store_true')
args = parser.parse_args()
def best_param(preds,targets,preserve):
    preds_eval = np.vstack([preds[i] for i in range(len(preds))])
    targets_eval = np.vstack([targets[i] for i in range(len(targets))])
    auc_score = []
    ap_score = []
    fdr_array=[]
    for i in range(103):
        precision, recall, thresholds = metrics.precision_recall_curve(targets_eval[:, i], preds_eval[:, i],
                                                                         pos_label=1)
        fdr = 1 - precision
        cutoff_index = next(i for i, x in enumerate(fdr) if x <= 0.5)
        fdr_at_cutoff = recall[cutoff_index]
        if not math.isnan(fdr_at_cutoff):
            fdr_array.append(np.nan_to_num(fdr_at_cutoff))
        auc_score.append(metrics.roc_auc_score(targets_eval[:, i], preds_eval[:, i]))
        ap_score.append(metrics.average_precision_score(targets_eval[:, i], preds_eval[:, i]))
    if preserve:
        np.save('graph.npy',np.array([auc_score,ap_score,fdr_array]))
    return np.mean(auc_score), np.mean(ap_score), np.mean(fdr_array)

class TrainDataset(Dataset):
    def __init__(self):
        cwd = '/nfs/turbo/umms-drjieliu/usr/zzh/chromGCN_data/GM12878/1000/'
        with open(cwd + 'processed_data/train_inputs.pickle', 'rb') as f:
            features = pickle.load(f)
        print('features load')
        with open(pwd + 'processed_data/train_inputs_revcom.pickle' , 'rb') as handle:
            revcom_fea=pickle.load(handle)
        with open(cwd + 'processed_data/train_labels.pickle', 'rb') as f:
            targets = pickle.load(f)
        print('targets load')
        keys = features.keys()
        temp_x=np.vstack([features[i] for i in keys])
        temp_rev_x = np.vstack([revcom_fea[i] for i in keys])
        temp_y = np.vstack([targets[i] for i in keys])
        self.x=torch.tensor(temp_x).float()
        self.rev_x = torch.tensor(temp_rev_x).float()
        self.y=torch.FloatTensor(temp_y)
        self.num= self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index],self.rev_x[index], self.y[index]
    def __len__(self):
        return self.num
class ValiDataset(Dataset):
    def __init__(self):
        cwd = '/nfs/turbo/umms-drjieliu/usr/zzh/chromGCN_data/GM12878/1000/'
        with open(cwd + 'processed_data/valid_inputs.pickle', 'rb') as f:
            features = pickle.load(f)
        print('features load')
        with open(pwd + 'processed_data/valid_inputs_revcom.pickle' , 'rb') as handle:
            revcom_fea=pickle.load(handle)
        with open(cwd + 'processed_data/valid_labels.pickle', 'rb') as f:
            targets = pickle.load(f)
        print('targets load')
        keys = features.keys()
        temp_x=np.vstack([features[i] for i in keys])
        temp_rev_x=np.vstack([revcom_fea[i] for i in keys])
        temp_y = np.vstack([targets[i] for i in keys])
        self.x=torch.tensor(temp_x).float()
        self.rev_x=torch.tensor(temp_rev_x).float()
        self.y=torch.FloatTensor(temp_y)
        self.num= self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index],self.rev_x[index], self.y[index]
    def __len__(self):
        return self.num
class TestDataset(Dataset):
    def __init__(self):
        cwd = '/nfs/turbo/umms-drjieliu/usr/zzh/chromGCN_data/GM12878/1000/'
        with open(cwd + 'processed_data/test_inputs.pickle', 'rb') as f:
            features = pickle.load(f)
        print('features load')
        with open(pwd + 'processed_data/test_inputs_revcom.pickle' , 'rb') as handle:
            revcom_fea=pickle.load(handle)
        with open(cwd + 'processed_data/test_labels.pickle', 'rb') as f:
            targets = pickle.load(f)
        print('targets load')
        keys = features.keys()
        temp_x=np.vstack([features[i] for i in keys])
        temp_rev_x = np.vstack([revcom_fea[i] for i in keys])
        temp_y = np.vstack([targets[i] for i in keys])
        self.x=torch.tensor(temp_x).float()
        self.rev_x = torch.tensor(temp_rev_x).float()
        self.y=torch.FloatTensor(temp_y)
        self.num= self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index],self.rev_x[index], self.y[index]
    def __len__(self):
        return self.num
model=Expecto(103,2000).to(device)
optimizer = optim.SGD(model.parameters(),weight_decay=1e-6, lr=0.25,momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.000001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8,verbose=True,factor=0.75)
loss_func = nn.BCEWithLogitsLoss()

print('starting training')
# training and validating
pwd = '/nfs/turbo/umms-drjieliu/usr/zzh/chromGCN_data/GM12878/1000/'

best_loss=1000
best_auc=0
best_ap=0
sums=0
if args.test:
    testloader = DataLoader(dataset=TestDataset(), batch_size=128, shuffle=False, num_workers=3)
    model.load_state_dict(torch.load('dcnnauc_rev_nd.pt'))
    model.eval()
    pred_test = []
    target_test = []
    for test_step, (test_batch_x, test_rev_x, test_batch_y) in enumerate(testloader):
        test_batch_x = test_batch_x.to(device)
        test_rev_x = test_rev_x.to(device)
        test_batch_y = test_batch_y.to(device)
        test_out, _ = model(test_batch_x)
        rev_test_out, _ = model(test_rev_x)
        test_loss = loss_func((test_out + rev_test_out) / 2, test_batch_y)
        pred_test.append(torch.sigmoid((test_out + rev_test_out) / 2).cpu().data.detach().numpy())
        target_test.append(test_batch_y.cpu().data.detach().numpy())
    auc_score, ap_score, fdr = best_param(pred_test, target_test,1)
    print('mean auc is %s, mean ap is %s, fdr is %s' % (auc_score,ap_score, fdr))
else:
    dataloader = DataLoader(dataset=TrainDataset(), batch_size=64, shuffle=True, num_workers=3)
    validloader = DataLoader(dataset=ValiDataset(), batch_size=128, shuffle=False, num_workers=3)
    for epoch in range(50):
        valid_losses = []
        train_losses=[]
        model.train()
        pred_train = []
        target_train = []
        for step, (train_batch_x, train_rev_x, train_batch_y) in enumerate(dataloader):
            t = time.time()
            train_batch_x = train_batch_x.to(device)
            train_rev_x=train_rev_x.to(device)
            train_batch_y = train_batch_y.to(device)

            out,_ = model(train_batch_x)
            rev,_ = model(train_rev_x)
            loss = loss_func((out+rev)/2, train_batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cur_loss=loss.item()
            train_losses.append(cur_loss)
            # pred_train.append(torch.sigmoid(out).cpu().data.detach().numpy())
            # target_train.append(train_batch_y.cpu().data.detach().numpy())
            if step %500 ==0:
                print("Epoch:", '%04d' % (epoch + 1), "step:", '%04d' % (step + 1),"train_loss=", "{:.7f}".format(cur_loss),
                      "time=", "{:.5f}".format(time.time() - t)
                      )
        train_loss = np.average(train_losses)
        print('training loss is %s'%train_loss)
        model.eval()
        pred_eval = []
        target_eval = []
        for valid_step, (valid_batch_x,valid_rev_x, valid_batch_y) in enumerate(validloader):
            valid_batch_x = valid_batch_x.to(device)
            valid_rev_x=valid_rev_x.to(device)
            valid_batch_y = valid_batch_y.to(device)

            val_out,_ = model(valid_batch_x)
            rev_val_out,_ = model(valid_rev_x)
            val_loss = loss_func((val_out+rev_val_out)/2, valid_batch_y)
            pred_eval.append(torch.sigmoid((val_out+rev_val_out)/2).cpu().data.detach().numpy())
            target_eval.append(valid_batch_y.cpu().data.detach().numpy())
            valid_losses.append(val_loss.item())
        auc_score, ap_score,fdr=best_param(pred_eval,target_eval)
        print('mean auc is %s, mean ap is %s, fdr is %s' % (auc_score, ap_score, fdr))
        valid_loss = np.average(valid_losses)
        print('valid Loss is %s'%valid_loss)
        sum=auc_score+ap_score+fdr
        if sum > sums:
            sums=sum
            torch.save(model.state_dict(), pwd + 'code/deepcnn_sum_rev_nd.pt')
            print('save sum model')
        if valid_loss< best_loss:
            best_loss=valid_loss
            torch.save(model.state_dict(), pwd + 'code/deepcnn_rev_nd.pt')
            print('save model')

        if auc_score>best_auc:
            best_auc=auc_score
            torch.save(model.state_dict(), pwd + 'code/dcnnauc_rev_nd.pt')
            print('best auc')
        if ap_score>best_ap:
            best_ap=ap_score
        #     torch.save(model.state_dict(), pwd + 'code/dcnnap_embed.pt')
            print('best ap')