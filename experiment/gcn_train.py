import numpy as np
import pickle
from sklearn import metrics
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time,argparse
import math
from chromgcn import ChromeGCN
from chromgcn_datasets import train_sets,test_sets,valid_sets
device = torch.device( 'cuda:0' if torch.cuda.is_available() else'cpu')
pwd = '/nfs/turbo/umms-drjieliu/usr/zzh/chromGCN_data/GM12878/1000/'
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
            np.save('gcn.npy', np.array([auc_score, ap_score, fdr_array]))
    return auc_score, ap_score, fdr_array
train_chr=[2,4,5,6,7,9,10,11,13,14,15,16,18,19,20,22]
valid_chr=[3,12,17]
test_chr=[1,8,21]

model=ChromeGCN(128,128,103).to(device)
optimizer = optim.SGD(model.parameters(),weight_decay=1e-6, lr=0.25,momentum=0.9)
loss_func = nn.BCEWithLogitsLoss()

best_loss=1000
best_auc=0
best_ap=0
if args.test:
    print('model test')
    model.load_state_dict(torch.load('gcnauc_ft.pt'))
    model.eval()
    test_input, test_input_rev, test_labels, test_adjs = test_sets().get_data()
    test_losses = []
    pred_test = []
    target_test = []
    for k in test_chr:
        t = time.time()
        test_batch_x = test_input[k].to(device)
        test_rev_x = test_input_rev[k].to(device)
        test_batch_y = test_labels[k].to(device)
        test_batch_adj = test_adjs[k].to(device)
        test_out = model(test_batch_x, test_batch_adj)
        test_rev_out = model(test_rev_x, test_batch_adj)
        test_loss = loss_func((test_out + test_rev_out) / 2, test_batch_y)
        pred_test.append(torch.sigmoid((test_out + test_rev_out) / 2).cpu().data.detach().numpy())
        target_test.append(test_batch_y.cpu().data.detach().numpy())
        test_losses.append(test_loss.item())
    auc_score, ap_score, fdr = best_param(pred_test, target_test,1)
    print('test auc is %s, test ap is %s, fdr is %s' % (np.mean(auc_score), np.mean(ap_score), np.mean(fdr)))
    print(np.mean(auc_score[:90]), np.mean(ap_score[:90]), np.mean(fdr[:90]))
    print(np.mean(auc_score[90:101]), np.mean(ap_score[90:101]), np.mean(fdr[90:101]))
    print(np.mean(auc_score[101:]), np.mean(ap_score[101:]), np.mean(fdr[101:]))

else:
    print('start training')
    train_input, train_input_rev, train_labels, train_adjs = train_sets().get_data()
    valid_input, valid_input_rev, valid_labels, valid_adjs = valid_sets().get_data()
    for epoch in range(1000):
        valid_losses = []
        train_losses=[]
        model.train()
        for k in train_chr:
            t = time.time()
            train_batch_x=train_input[k].to(device)
            train_rev_x = train_input_rev[k].to(device)
            train_batch_y = train_labels[k].to(device)
            train_batch_adj=train_adjs[k].to(device)
            out = model(train_batch_x,train_batch_adj)
            rev_out = model(train_rev_x, train_batch_adj)
            loss = loss_func((out+rev_out)/2, train_batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cur_loss = loss.item()
            train_losses.append(cur_loss)
        train_loss = np.average(train_losses)
        print('training loss is %s' % train_loss)
        if epoch<400:
            print(str(epoch)+' heatup')
            continue
        model.eval()
        pred_eval = []
        target_eval = []
        for k in valid_chr:
            t = time.time()
            valid_batch_x=valid_input[k].to(device)
            valid_rev_x = valid_input_rev[k].to(device)
            valid_batch_y = valid_labels[k].to(device)
            valid_batch_adj = valid_adjs[k].to(device)
            val_out = model(valid_batch_x,valid_batch_adj)
            val_rev_out = model(valid_rev_x, valid_batch_adj)
            val_loss = loss_func((val_out+val_rev_out)/2, valid_batch_y)
            pred_eval.append(torch.sigmoid((val_out+val_rev_out)/2).cpu().data.detach().numpy())
            target_eval.append(valid_batch_y.cpu().data.detach().numpy())
            valid_losses.append(val_loss.item())
        auc_score, ap_score, fdr = best_param(pred_eval, target_eval,0)
        print('mean auc is %s, mean ap is %s, fdr is %s' % (np.mean(auc_score), np.mean(ap_score),np.mean(fdr)))
        valid_loss = np.average(valid_losses)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), pwd + 'finetune_code/gcn.pt')
            print('save model')
        if np.mean(auc_score) > best_auc:
            best_auc = np.mean(auc_score)
            torch.save(model.state_dict(), pwd + 'finetune_code/gcnauc_ft.pt')
            print('save best auc')