import numpy as np
from sklearn import metrics
import multiprocessing as mp
from torch.utils.data import DataLoader
import torch,time,argparse
from pretrain_layer import Expecto,DeepSEA,DanQ
import torch.optim as optim
import torch.nn as nn
from pretrain_dataset import TrainDataset,ValidDataset,TestDataset

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.5, help='Learning rate.')
parser.add_argument('--seq_length', type=int, default=1000, help='input sequence length')
parser.add_argument('--length', type=int, default=2600, help='length of hidden representation')
parser.add_argument('--pre_model', type=str, choices=['deepsea','expecto','danq']
                    , default='expecto',help='pre-train models')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--label_size', type=int, default=2583)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--load_model', default=False, action='store_true',help='load trained model')
parser.add_argument('--test', default=False, action='store_true',help='model testing')

args = parser.parse_args()
if torch.cuda.device_count()>1:
    mode='multi'
else:
    mode='single'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
def best_param(preds,targets,preserve):
    preds_eval = np.vstack([preds[i] for i in range(len(preds))])
    targets_eval = np.vstack([targets[i] for i in range(len(targets))])
    pool = mp.Pool(32)
    result = pool.map_async(cal_auc, ((targets_eval[:, i], preds_eval[:, i]) for i in range(preds_eval.shape[1])))
    pool.close()
    pool.join()
    r = np.array(result.get())
    r=r[~np.isnan(r)].reshape(-1,2)
    if preserve:
        np.save('compare_metrics/%s.npy'%(args.pre_model),r)
    return np.mean(r[:,0]),np.mean(r[:,1])

train_chr=[1,4,5,6,7,9,10,11,13,14,15,16,17,18,19,20,22]
valid_chr=[3,12]
test_chr=[2,8,21]
if args.pre_model=='deepsea':
    model=DeepSEA(args.label_size,args.seq_length,args.length)
elif args.pre_model=='expecto':
    model=Expecto(args.label_size,args.seq_length,args.length)
else:
    model=DanQ(args.label_size,args.seq_length,args.length)
if mode=='multi':
    model=nn.DataParallel(model)
model.to(device)

if args.load_model:
    print('load models......')
    model.load_state_dict(torch.load('models/%s_auc_%s.pt'%(args.pre_model,args.length)))
optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9,weight_decay=1e-6)
# schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1, patience=5,verbose=1)

loss_func = nn.BCEWithLogitsLoss()
if args.test:
    print('model test')
    testloader = DataLoader(dataset=TestDataset(), batch_size=256, shuffle=False, num_workers=2)
    model.load_state_dict(torch.load('models/%s_auc_%s.pt'%(args.pre_model,args.length)))
    model.eval()
    pred_eval = []
    target_eval = []
    test_losses=[]
    for step, (test_batch_x, test_batch_y) in enumerate(testloader):
        t = time.time()
        test_batch_x = test_batch_x.float().to(device)
        test_batch_y = test_batch_y.to(device)
        out, _ = model(test_batch_x)
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
    test_loss = np.average(test_losses)
    print('test Loss is %s' % test_loss)
    auc_score, ap_score =best_param(pred_eval, target_eval,1)
    print('mean auc is %s, mean ap is %s' % (auc_score, ap_score))

else:
    print('model training')
    trainloader = DataLoader(dataset=TrainDataset(), batch_size=args.batchsize, shuffle=True, num_workers=2)
    validloader = DataLoader(dataset=ValidDataset(), batch_size=256, shuffle=False, num_workers=2)
    best_loss=1000
    best_auc=0
    best_ap=0
    for epoch in range(args.epochs):
        valid_losses = []
        train_losses = []
        test_losses = []
        model.train()
        for step, (train_batch_x, train_batch_y) in enumerate(trainloader):
            t = time.time()
            train_batch_x=train_batch_x.float().to(device)
            train_batch_y=train_batch_y.to(device)
            out,_ = model(train_batch_x)
            loss=loss_func(out,train_batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cur_loss = loss.item()
            train_losses.append(cur_loss)
            if step %30000 ==0:
                print("Epoch:", '%04d' % (epoch + 1), "step:", '%04d' % (step + 1),"train_loss=", "{:.7f}".format(cur_loss),
                      "time=", "{:.5f}".format(time.time() - t)
                      )
        train_loss = np.average(train_losses)
        print('training loss is %s' % train_loss)
        model.eval()
        pred_eval = []
        target_eval = []
        auc_record = []
        for step, (valid_batch_x, valid_batch_y) in enumerate(validloader):
            t = time.time()
            valid_batch_x = valid_batch_x.float().to(device)
            valid_batch_y = valid_batch_y.to(device)
            out, _ = model(valid_batch_x)
            loss = loss_func(out, valid_batch_y)
            cur_loss = loss.item()
            valid_losses.append(cur_loss)
            pred_eval.append(torch.sigmoid(out).cpu().data.detach().numpy())
            target_eval.append(valid_batch_y.cpu().data.detach().numpy())
        auc_score, ap_score= best_param(pred_eval, target_eval,0)
        print('mean auc is %s, mean ap is %s' % (auc_score, ap_score))
        valid_loss = np.average(valid_losses)
        # schedule.step(valid_loss)
        print('valid Loss is %s' % valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(),'models/%s_loss_%s.pt'%(args.pre_model,args.length))
            print('save model')
        if auc_score > best_auc:
            auc_record = []
            best_auc = auc_score
            torch.save(model.state_dict(),'models/%s_auc_%s.pt'%(args.pre_model,args.length))
            print('save best auc')
        else:
            auc_record.append(auc_score)
        if len(auc_record) >= 4:
            print('Early stop')
            break