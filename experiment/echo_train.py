import numpy as np
import pickle
from sklearn import metrics
import torch
import torch.optim as optim
import torch.nn as nn
import math
import torch.utils.data as Data
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
from echo_layer import ECHO
from neighborsets import TrainDataset,ValiDataset,TestDataset
device = torch.device( 'cuda:0' if torch.cuda.is_available() else'cpu')
pwd = '/nfs/turbo/umms-drjieliu/usr/zzh/chromGCN_data/GM12878/1000/'
def best_param(preds,targets):
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
    return np.mean(auc_score), np.mean(ap_score), np.mean(fdr_array)


model=ECHO(103,30,10).to(device)
optimizer = optim.SGD(model.parameters(),weight_decay=1e-6, lr=0.25,momentum=0.9)
# schedule = optim.lr_scheduler.StepLR(optimizer,step_size=15, gamma=0.7)
# optimizer = optim.Adam(model.parameters(), lr=0.1)
# schedule=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10,verbose=True)
loss_func = nn.BCEWithLogitsLoss()
dataloader=DataLoader(dataset=TrainDataset(),batch_size=64,shuffle=True, num_workers=2)
validloader=DataLoader(dataset=ValiDataset(),batch_size=512,shuffle=False, num_workers=2)
testloader=DataLoader(dataset=TestDataset(),batch_size=512,shuffle=False, num_workers=2)
best_loss=1000
best_auc=0
best_ap=0
best_fdr=0
print('start training')
for epoch in range(60):
    valid_losses = []
    train_losses=[]
    test_losses=[]
    model.train()
    for step, (train_batch_x,train_batch_x1,train_rev_x,train_rev_x1, train_batch_y) in enumerate(dataloader):
        t = time.time()
        train_batch_x = train_batch_x.to(device)
        train_batch_x1 = train_batch_x1.to(device)
        train_rev_x=train_rev_x.to(device)
        train_rev_x1=train_rev_x1.to(device)
        train_batch_y = train_batch_y.to(device)
        out = model(train_batch_x,train_batch_x1,train_rev_x,train_rev_x1)
        loss = loss_func(out, train_batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_loss=loss.item()
        train_losses.append(cur_loss)
        if step %500 ==0:
            print("Epoch:", '%04d' % (epoch + 1), "step:", '%04d' % (step + 1),"train_loss=", "{:.7f}".format(cur_loss),
                  "time=", "{:.5f}".format(time.time() - t)
                  )
    train_loss = np.average(train_losses)
    print('training loss is %s'%train_loss)

    model.eval()
    pred_eval = []
    target_eval = []
    for valid_step, (valid_batch_x,valid_batch_x1,valid_rev_x,valid_rev_x1, valid_batch_y) in enumerate(validloader):
        valid_batch_x = valid_batch_x.to(device)
        valid_batch_y = valid_batch_y.to(device)
        valid_batch_x1 = valid_batch_x1.to(device)
        valid_rev_x=valid_rev_x.to(device)
        valid_rev_x1=valid_rev_x1.to(device)
        val_out = model(valid_batch_x,valid_batch_x1,valid_rev_x, valid_rev_x1)

        val_loss = loss_func(val_out, valid_batch_y)
        pred_eval.append(torch.sigmoid(val_out).cpu().data.detach().numpy())
        target_eval.append(valid_batch_y.cpu().data.detach().numpy())
        valid_losses.append(val_loss.item())
    auc_score, ap_score,fdr=best_param(pred_eval,target_eval)
    print('mean auc is %s, mean ap is %s, fdr is %s' % (auc_score, ap_score,fdr))
    valid_loss = np.average(valid_losses)
    # schedule.step()
    print('valid Loss is %s' % valid_loss)
    with open('valid_metrics_graph_rev_nd.txt', 'a') as f:
        f.write(str(epoch)+','+str(valid_loss)+','+str(auc_score)+','+str(ap_score)+','+str(fdr)+'\n')

    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), pwd + 'finetune_code/neighbor.pt')
        print('save model')
    if auc_score > best_auc:
        best_auc = auc_score
        torch.save(model.state_dict(), pwd + 'finetune_code/neighborauc_ft.pt')
        print('save best auc')

    pred_test = []
    target_test= []
    for test_step, (test_batch_x,test_batch_x1,test_rev_x,test_rev_x1, test_batch_y) in enumerate(testloader):
        test_batch_x = test_batch_x.to(device)
        test_batch_x1 = test_batch_x1.to(device)

        test_rev_x = test_rev_x.to(device)
        test_rev_x1 = test_rev_x1.to(device)
        test_batch_y = test_batch_y.to(device)

        test_out = model(test_batch_x,test_batch_x1,test_rev_x, test_rev_x1)
        test_loss = loss_func(test_out, test_batch_y)
        pred_test.append(torch.sigmoid(test_out).cpu().data.detach().numpy())
        target_test.append(test_batch_y.cpu().data.detach().numpy())
        test_losses.append(test_loss.item())
    auc_score, ap_score,fdr = best_param(pred_test, target_test)
    print('test auc is %s, test ap is %s, fdr is %s' % (auc_score, ap_score,fdr))
    test_loss = np.average(test_losses)
    print('test Loss is %s' % test_loss)
    with open('test_metrics_graph_rev_nd.txt', 'a') as f:
        f.write(str(epoch)+','+str(test_loss)+','+str(auc_score)+','+str(ap_score)+','+str(fdr)+'\n')