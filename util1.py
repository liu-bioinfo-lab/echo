from pretrain_layer import Expecto
from graph_layer import ECHO
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch,time,pickle
from sklearn.preprocessing import normalize
import numpy as np

with open('chromatin_feature.pickle','rb') as f:
  chromatin_feature=pickle.load(f)
def find_tf_idx(inspect_tf):
  tf_indices=[]
  for k in chromatin_feature.keys():
    if inspect_tf.lower() == chromatin_feature[k].split('\t')[0].lower():
      tf_indices.append(k)
  return np.sort(tf_indices)

def find_binding_locs(inspect_tf,labels,chr):
  tf_indices=find_tf_idx(inspect_tf)
  bind_locs=np.where(np.sum(labels[chr][:,tf_indices],1)>0)[0]
  return bind_locs

one_hot_dic={'a':0,'c':1,'g':2,'t':3}
def encoding(text):
    text=text.lower()
    encode_text=np.zeros((4,len(text)))
    for idx in range(len(text)):
        encode_text[one_hot_dic[text[idx]],idx]=1
    return encode_text
def generate_inputs(input_sample_poi,chr,ref_genome):
    temps = []
    for idx in input_sample_poi[chr]:
      temp = []
      for i in range(-2, 3):
        temp.append(encoding(ref_genome[chr][idx + 200 * i]))
      temp1 = np.hstack([i for i in temp])
      temps.append(temp1)
    input_sequence=np.array(temps,dtype=np.int8)
    return input_sequence

class Dataset2(Dataset):
    def __init__(self,neighs,chr,binding_locs):
        # binding_locs=find_binding_locs(inspect_tf,labels,chr)
        self.x_idx=neighs[chr][binding_locs,:]
        self.num = self.x_idx.shape[0]
    def __getitem__(self, index):
        return self.x_idx[index]
    def __len__(self):
        return self.num


def filter_sequence(inputs, neighs, labels,input_sample_poi, inspect_tf,contact_threshold=0.4, score_threshold=0.3,seq_threshold=300):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

    def get_diag(matrix):
        temp = []
        for i in range(matrix.shape[0]):
            temp.append(np.diag(matrix[i, :, :]))
        return np.array(temp)

    def remove_overlapping_sequence(central_seq_idx, chr, neighbor_seqs, grad_atts):
        central_seq = input_sample_poi[chr][central_seq_idx]
        pad_idx = input_sample_poi[chr].shape[0]
        for i in range(neighbor_seqs.shape[0]):
            if neighbor_seqs[i] == pad_idx:
                grad_atts[i] = 0
                continue
            if np.abs(input_sample_poi[chr][neighbor_seqs[i]] - central_seq) <= 400:
                grad_atts[i] = 0
        return grad_atts

    num_sequences = 0
    sequence_grad = []
    sequence_input = []
    tf_indices = find_tf_idx(inspect_tf)
    print(tf_indices)
    for chr in inputs.keys():
        pad_idx = inputs[chr].shape[0]
        binding_locs = find_binding_locs(inspect_tf, labels, chr)
        print(binding_locs.shape)
        dataloader = DataLoader(dataset=Dataset2(neighs, chr, binding_locs), batch_size=1, shuffle=True)
        input_fea = inputs[chr]
        input_fea = np.vstack((input_fea, np.zeros((1, 4, 1000), dtype=np.int8)))
        input_fea = torch.tensor(input_fea).float()
        for step, (test_x_idx) in enumerate(dataloader):
            xidx = test_x_idx.flatten()
            central_seq_idx = xidx[55]
            input_xfea = input_fea[xidx, :, :].to(device)
            input_xfea.requires_grad = True
            _, xfea = model(input_xfea)
            pad_indices = np.where(xidx == pad_idx)[0]
            if not pad_indices.shape[0]:
                paddings = torch.zeros(pad_indices.shape[0], 2600).to(device)
                xfea[pad_indices, :] = paddings
            xfea = xfea.reshape(test_x_idx.shape[0], 61, 2600)
            att1 = torch.eye(50, requires_grad=True) \
                .reshape(1, 50, 50).float().to(device)
            att2 = torch.eye(11, requires_grad=True) \
                .reshape(1, 11, 11).float().to(device)
            xfea1 = xfea[:, :50, :]
            xfea2 = xfea[:, 50:61, :]
            xfea1 = torch.matmul(att1, xfea1)
            xfea2 = torch.matmul(att2, xfea2)
            out = graph_model(xfea1, xfea2)

            tf_binding_indices = np.where(labels[chr][central_seq_idx, tf_indices].toarray().squeeze() > 0)[0]
            if torch.mean(torch.sigmoid(out)[:, tf_indices[tf_binding_indices]]) < score_threshold:
                continue
            att1.retain_grad()
            att2.retain_grad()
            gout = torch.sum(out[:, tf_indices[tf_binding_indices]])
            gout.backward()
            grad_att1 = get_diag(att1.grad.data.cpu().detach().numpy())
            grad_att2 = get_diag(att2.grad.data.cpu().detach().numpy())
            grad_atts = np.hstack((grad_att1, grad_att2))
            grad_atts = normalize(np.absolute(grad_atts), norm='max').squeeze()
            neighbor_seqs = neighs[chr][central_seq_idx, :]

            grad_atts = remove_overlapping_sequence(central_seq_idx, chr, neighbor_seqs, grad_atts)

            temp = {}
            temp_idx = np.where(grad_atts > contact_threshold)[0]
            if not temp_idx.shape[0]:
                continue
            for i in temp_idx:
                if neighbor_seqs[i] in temp.keys():
                    if grad_atts[i] > grad_atts[temp[neighbor_seqs[i]]]:
                        temp[neighbor_seqs[i]] = i
                else:
                    temp[neighbor_seqs[i]] = i
            filter_seq = list(temp.values())
            grads = input_xfea.grad.data.cpu().detach().numpy().reshape(61, 4, 1000)[np.array(filter_seq), :, :]
            input_seq = input_xfea.cpu().detach().numpy().reshape(61, 4, 1000)[np.array(filter_seq), :, :]
            num_sequences += len(filter_seq)
            sequence_grad.append(grads)
            sequence_input.append(input_seq)
            print('%s sequences are found' % num_sequences)
            if num_sequences > seq_threshold:
                return np.vstack(sequence_grad), np.vstack(sequence_input)
    return np.vstack(sequence_grad), np.vstack(sequence_input)