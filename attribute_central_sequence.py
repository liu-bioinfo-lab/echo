from scipy import sparse
import torch,pickle,argparse
from pretrain_layer import Expecto
from graph_layer import ECHO
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--tf', type=str, default='JUND')
parser.add_argument('--filter_score', type=float, default=0.5)
args = parser.parse_args()

device = torch.device( 'cuda:0' if torch.cuda.is_available() else'cpu')
print(device)
model = Expecto(2583, 1000, 2600)
model.to(device)
model.load_state_dict(torch.load('models/expecto_auc_2600.pt', map_location=device))
model.eval()
graph_model = ECHO(2583, 50, 10)
graph_model.to(device)
graph_model.load_state_dict(
        torch.load('models/echo_auc_expecto_2600_50_10.pt', map_location=device))
graph_model.eval()


with open('chromatin_feature.pickle','rb') as f:
    chromatin_feature=pickle.load(f)
def find_tf_idx(inspect_tf):
    tf_indices=[]
    for k in chromatin_feature.keys():
        if inspect_tf.lower() == chromatin_feature[k].split('\t')[0].lower():
            tf_indices.append(k)
    if not tf_indices:
        raise ValueError("no such chromatin feature")
    return np.sort(tf_indices)

tf_sets=find_tf_idx(args.tf)
print(tf_sets)

inspect_chr=2
# load labels
labels=sparse.load_npz('labels/ihec_labels/chr%s.npz'%inspect_chr).toarray()
tfs_label=np.sum(labels[:,tf_sets],1)
tfs_locs=np.where(tfs_label>0)[0]
print(tfs_locs.shape)

# load neighbor set
with open('neighbor_list/neighbor_list_50_5.pickle','rb') as f:
    adjs=pickle.load(f)
adj_matrix=adjs[inspect_chr]

# load input DNA sequences
inputs=np.load('inputs/chr%s.npy'%inspect_chr)
print(inputs.shape,labels.shape)
inputs=np.vstack((inputs,np.zeros((1,4,args.seq_length),dtype=np.int8)))

gradient=[]
input_poi=[]
for idx in tfs_locs:
    model.eval()
    graph_model.eval()
    tf_idx = np.where(labels[idx,tf_sets] > 0)[0]
    tf_idx = tf_sets[tf_idx]
    input_fea = torch.tensor(inputs[adj_matrix[idx, :], :, :]).squeeze().float().to(device)
    input_fea.requires_grad = True
    _, xfea = model(input_fea)
    xfea1 = xfea[:args.k_adj, :].unsqueeze(0)
    xfea2 = xfea[args.k_adj:, :].unsqueeze(0)
    out = graph_model(xfea1, xfea2, None)
    tfs_out = torch.mean(torch.sigmoid(out[:, tf_idx]), 1)
    if tfs_out > args.filter_score:
        print(idx,tfs_out.item())
        out = torch.sum(out[:, tf_idx], 1)
        out.backward()
        grads = input_fea.grad.data.cpu().detach().numpy()
        gradient.append(grads[-6, :, :])
        input_poi.append(input_fea[-6, :, :].cpu().detach().numpy())
    # collect attribution scores of 200 central sequence
    if len(gradient) >= 200:
        print('stopping')
        break
gradient=np.vstack([np.expand_dims(i, axis=0) for i in gradient])
input_seqs=np.vstack([np.expand_dims(i, axis=0) for i in input_poi])
print(gradient.shape,input_seqs.shape)
np.save('data/%s_grad.npy'%args.tf,gradient)
np.save('data/%s_input.npy'%args.tf,input_poi)
