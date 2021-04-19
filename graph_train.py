import numpy as np
from sklearn import metrics
import multiprocessing as mp
from torch.utils.data import DataLoader
import torch,time,argparse
from graph_layer import ECHO
import torch.optim as optim
import torch.nn as nn
import pickle
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
parser.add_argument('--load_model', default=False, action='store_true',help='load trained model')
parser.add_argument('--test', default=False, action='store_true',help='model testing')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dir_path='/nfs/turbo/umms-drjieliu/usr/zzh/deepchrom/'
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
        np.save('models/echo_%s.npy'%(args.pre_model),r)
    return np.mean(r[:,0]),np.mean(r[:,1])

if torch.cuda.device_count()>1:
    mode='multi'

else:
    mode='single'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(mode+' GPU training')
train_chr=[1,4,5,6,7,9,10,11,13,14,15,16,17,18,19,20,22]
valid_chr=[3,12]
test_chr=[2,8,21]
dir_path='/nfs/turbo/umms-drjieliu/usr/zzh/deepchrom/'
with open(dir_path+'hidden_feature/hidden_auc_%s_%s.pickle'%(args.pre_model,args.pre_length), 'rb') as f:
    hidden_feas=pickle.load(f)

print('inputs load finished')
graph_model=ECHO(args.label_size,args.k_adj,args.k_neigh)

if mode=='multi':
    graph_model=nn.DataParallel(graph_model)
graph_model.to(device)
if args.load_model:
    print('load models')
    graph_model.load_state_dict(torch.load(dir_path+'models/echo_auc_%s_%s_%s_%s.pt' %
        (args.pre_model, args.pre_length, args.k_adj, args.k_neigh)))

optimizer = optim.SGD(graph_model.parameters(), lr=args.lr,momentum=0.9,weight_decay=1e-6)
loss_func = nn.BCEWithLogitsLoss()

if args.test:
    print('model test')
    graph_model.load_state_dict(torch.load(dir_path+'models/echo_auc_%s_%s_%s_%s.pt' %
                           (args.pre_model, args.pre_length, args.k_adj, args.k_neigh)))

    test_inputs = np.vstack([np.vstack((hidden_feas[chr], np.zeros((1, args.pre_length), dtype=np.float32)))
                             for chr in test_chr])
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
    test_loss = np.average(test_losses)
    print('test Loss is %s' % test_loss)
    auc_score, ap_score=best_param(pred_eval, target_eval,1)
    print('mean auc is %s, mean ap is %s' % (auc_score, ap_score))
else:
    print('model training')
    # add dummy sequence
    train_inputs = np.vstack([np.vstack((hidden_feas[chr], np.zeros((1, args.pre_length), dtype=np.float32)))
                              for chr in train_chr])
    train_inputs = torch.tensor(train_inputs)
    print(train_inputs.shape)
    valid_inputs = np.vstack([np.vstack((hidden_feas[chr], np.zeros((1, args.pre_length), dtype=np.float32)))
                              for chr in valid_chr])
    valid_inputs = torch.tensor(valid_inputs)
    print(valid_inputs.shape)
    trainloader = DataLoader(dataset=TrainDataset(), batch_size=args.batchsize, shuffle=True,
                             num_workers=2)
    validloader = DataLoader(dataset=ValidDataset(), batch_size=args.batchsize, shuffle=False,
                             num_workers=2)
    best_loss = 1000
    best_auc = 0
    best_ap = 0
    for epoch in range(args.epochs):
        valid_losses = []
        train_losses = []
        test_losses = []
        graph_model.train()
        for step, (train_x_idx, train_batch_y) in enumerate(trainloader):
            t = time.time()
            xidx =train_x_idx.flatten()
            xfea = train_inputs[xidx, :].to(device)
            train_batch_y = train_batch_y.to(device)
            xfea=xfea.reshape(train_batch_y.shape[0],args.k_adj+args.k_neigh+1,args.pre_length)
            xfea1=xfea[:,:args.k_adj,:]
            xfea2=xfea[:,args.k_adj:args.k_adj+args.k_neigh+1,:]
            out = graph_model(xfea1, xfea2)
            loss=loss_func(out,train_batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cur_loss = loss.item()
            train_losses.append(cur_loss)
            if step %10000 ==0:
                print("Epoch:", '%04d' % (epoch + 1), "step:", '%04d' % (step + 1),"train_loss=", "{:.7f}".format(cur_loss),
                      "time=", "{:.5f}".format(time.time() - t)
                      )
        train_loss = np.average(train_losses)
        print('training loss is %s' % train_loss)
        graph_model.eval()
        pred_eval = []
        target_eval = []
        auc_record = []
        for step, (valid_x_idx, valid_batch_y) in enumerate(validloader):
            t = time.time()
            xidx = valid_x_idx.flatten()
            xfea= valid_inputs[xidx, :].to(device)
            valid_batch_y = valid_batch_y.to(device)
            xfea=xfea.reshape(valid_batch_y.shape[0],args.k_adj+args.k_neigh+1,args.pre_length)
            xfea1=xfea[:,:args.k_adj,:]
            xfea2=xfea[:,args.k_adj:args.k_adj+args.k_neigh+1,:]
            out = graph_model(xfea1, xfea2)
            loss=loss_func(out,valid_batch_y)
            cur_loss = loss.item()
            valid_losses.append(cur_loss)
            pred_eval.append(torch.sigmoid(out).cpu().data.detach().numpy())
            target_eval.append(valid_batch_y.cpu().data.detach().numpy())
        auc_score, ap_score= best_param(pred_eval, target_eval,0)
        print('mean auc is %s, mean ap is %s' % (auc_score, ap_score))
        valid_loss = np.average(valid_losses)
        # schedule.step()
        print('valid Loss is %s' % valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(graph_model.state_dict(), dir_path+'models/echo_loss_%s_%s_%s_%s.pt' %
                           (args.pre_model, args.pre_length, args.k_adj, args.k_neigh))
            print('save model')
        if auc_score > best_auc:
            auc_record = []
            best_auc = auc_score
            torch.save(graph_model.state_dict(), dir_path+'models/echo_auc_%s_%s_%s_%s.pt' %
                           (args.pre_model, args.pre_length, args.k_adj, args.k_neigh))
            print('save best auc')
        else:
            auc_record.append(auc_score)
        if len(auc_record)>=4:
            print('Early stop')
            break