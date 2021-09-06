import torch,pickle,time
import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from graph_layer import ECHO
from pretrain_layer import Expecto
class Dataset2(Dataset):
    def __init__(self,neighs,chr):

        self.x_idx=neighs[chr]
        self.num = self.x_idx.shape[0]
    def __getitem__(self, index):
        return self.x_idx[index]
    def __len__(self):
        return self.num



def echo_attribute(inputs,neighs,threshold):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = Expecto(2583, 1000, 2600)
    model.to(device)
    model.load_state_dict(torch.load('models/expecto_auc_2600.pt', map_location=device))
    model.eval()
    graph_model = ECHO(2583, 50, 10)
    graph_model.to(device)
    graph_model.load_state_dict(torch.load('models/echo_auc_expecto_2600_50_10.pt', map_location=device))
    graph_model.eval()
    def normalize(data):
        data = np.abs(data) / np.amax(np.abs(data))
        return data
    def get_diag(matrix):
        temp = []
        for i in range(matrix.shape[0]):
            temp.append(np.diag(matrix[i, :, :]))
        return np.array(temp)

    outputs = {}
    att_contacts = {}
    att_neighs = {}
    for chr in inputs.keys():
        att_contacts[chr]={}
        att_neighs[chr]={}
    def process_data(pad_idx, neighbor_idx, grad_seq, grad_atts, chr):
        centrals = neighbor_idx[:, -6]
        grad_atts=normalize(grad_atts)
        for idx in range(centrals.shape[0]):
            central_idx = centrals[idx].item()
            att_contacts[chr][central_idx]={}
            att_neighs[chr][central_idx] = {}
            for nidx in range(neighbor_idx.shape[1]):
                if neighbor_idx[idx, nidx] == pad_idx:
                    continue
                if neighbor_idx[idx, nidx] not in att_contacts[chr][central_idx].keys():
                    att_contacts[chr][central_idx][neighbor_idx[idx, nidx].item()]=grad_atts[idx,nidx].item()
                    att_neighs[chr][central_idx][neighbor_idx[idx, nidx].item()] = grad_seq[idx, nidx,:,:]
                else:
                    if grad_atts[idx,nidx]>att_contacts[chr][central_idx][neighbor_idx[idx, nidx]]:
                        att_contacts[chr][central_idx][neighbor_idx[idx, nidx].item()] = grad_atts[idx, nidx].item()
                        att_neighs[chr][central_idx][neighbor_idx[idx, nidx].item()] = grad_seq[idx, nidx, :, :]
    for chr in inputs.keys():
        tempo = []
        pad_idx = inputs[chr].shape[0]
        testloader = DataLoader(dataset=Dataset2(neighs,chr), batch_size=1, shuffle=False, num_workers=2)
        input_fea = inputs[chr]
        input_fea = np.vstack((input_fea, np.zeros((1, 4, 1000), dtype=np.int8)))
        input_fea = torch.tensor(input_fea).float()
        for step, (test_x_idx) in enumerate(testloader):
            t = time.time()
            xidx = test_x_idx.flatten()
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
            att1.retain_grad()
            att2.retain_grad()
            indices = (torch.sigmoid(out)[:, :882] > threshold).to(device)
            if torch.sum(indices)<1:
                central_idx=test_x_idx.numpy()[0,-6]
                att_contacts[chr][central_idx] = {}
                att_contacts[chr][central_idx]['None'] = 1
                att_neighs[chr][central_idx] = {}
                att_neighs[chr][central_idx]['None'] = 1
                tempo.append(torch.sigmoid(out).cpu().data.detach().numpy())
                if step % 1000 == 0:
                    print("step:", '%04d' % (step + 1), "time=", "{:.5f}".format(time.time() - t)
                          )
                continue
            gout = out[:, :882] * indices
            gout = torch.sum(gout)
            gout.backward()
            grad_att1 = get_diag(att1.grad.data.cpu().detach().numpy())
            grad_att2 = get_diag(att2.grad.data.cpu().detach().numpy())
            grad_atts=np.hstack((grad_att1, grad_att2))
            grad_seq = (input_xfea * input_xfea.grad.data).cpu().detach().numpy().reshape(test_x_idx.shape[0], 61, 4,
                                                                                          1000)
            process_data(pad_idx, test_x_idx.numpy(), np.abs(grad_seq), grad_atts, chr)
            tempo.append(torch.sigmoid(out).cpu().data.detach().numpy())
            if step % 1000 == 0:
                print("step:", '%04d' % (step + 1), "time=", "{:.5f}".format(time.time() - t)
                      )
        outputs[chr] = np.vstack(tempo)
    return outputs,att_neighs,att_contacts

from pyliftover import LiftOver
def process_bedfile(ATAC_file,version):
    atac_align={}
    lo = LiftOver('hg19', 'hg38')
    if version not in ['hg19', 'hg38']:
        raise ValueError('Invalid reference genome version Expected one of: %s'%(['hg19', 'hg38']))
    with open(ATAC_file,'r') as f:
        for line in f:
            content=line.strip().split('\t')
            try:
                chromosome=int(content[0][3:])
                if chromosome not in atac_align.keys():
                    atac_align[chromosome]=set()
                if version == 'hg19':
                    s = lo.convert_coordinate('chr%s' % chromosome, int(content[1]))[0][1]
                    e = lo.convert_coordinate('chr%s' % chromosome, int(content[2]))[0][1]
                else:
                    s=int(content[1])
                    e=int(content[2])
            except Exception:
                continue
            start=s//200*200
            end =(e//200+1)*200 if e%200 else e//200*200
            for loci in range(start,end,200):
                atac_align[chromosome].add(loci)
    for i in atac_align.keys():
        temp=np.sort(list(atac_align[i]))
        atac_align[i]=temp
    return atac_align

one_hot_dic={'a':0,'c':1,'g':2,'t':3}
def encoding(text):
    text=text.lower()
    encode_text=np.zeros((4,len(text)))
    for idx in range(len(text)):
        encode_text[one_hot_dic[text[idx]],idx]=1
    return encode_text


def generate_input(ATAC_file,version='hg38'):
    atac_align=process_bedfile(ATAC_file,version)
    with open('echo_data/ref_genome_200bp.pickle', 'rb') as f:
        ref_genome = pickle.load(f)
    inputs = {}
    for chr in atac_align.keys():
        input_sequence = []
        temps = []
        for idx in atac_align[chr]:
            temp = []
            try:
                for i in range(-2, 3):
                    temp.append(encoding(ref_genome[chr][idx + 200 * i]))
            except Exception:
                continue
            temp1 = np.hstack([i for i in temp])
            temps.append(temp1)
            input_sequence.append(idx)
        atac_align[chr] = np.array(input_sequence)
        inputs[chr] = np.array(temps,dtype=np.int8)
        print(atac_align[chr].shape, inputs[chr].shape)
    return inputs,atac_align

def build_graph(input_samples,datas,chr,cl):
    sample_dic={}
    for i in range(input_samples[chr].shape[0]):
        sample_dic[input_samples[chr][i]]=i
    if cl =='HFF':
        ratio=[1.885, 1.952, 1.961, 1.931, 1.914, 1.949, 1.783, 1.876, 1.772, 1.87,
       1.811, 1.909, 1.863, 1.762, 1.9, 1.633, 1.666, 1.813, 1.512, 1.689, 1.682, 1.384]
    else:
        ratio=[1 for _ in range(22)]
    row = []
    col = []
    vals = []
    for bin1 in input_samples[chr]:
        try:
            keys=list(datas[bin1].keys())
        except Exception:
            continue
        for k in keys:
            idx1=sample_dic[bin1]
            try:
                idx2=sample_dic[k]
            except Exception:
                continue
            row.extend([idx1,idx2])
            col.extend([idx2,idx1])
            value=np.round(datas[bin1][k] / ratio[chr - 1], 2)
            vals.extend([value,value])
    lens=input_samples[chr].shape[0]
    cmap = csr_matrix((np.array(vals), (np.array(row), np.array(col))), shape=(lens, lens))
    return cmap
def maximum (A, B):
    BisBigger = A-B
    BisBigger.data = np.where(BisBigger.data < 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)

def generate_contact_map(input_sample_poi):
    contact_maps = {}
    for chr in input_sample_poi.keys():
        with open('echo_data/hESC_chr%s.pickle' % chr, 'rb') as f:
            hesc_maps = pickle.load(f)
        with open('echo_data/HFF_chr%s.pickle' % chr, 'rb') as f:
            HFF_maps = pickle.load(f)
        hesc_cmap = build_graph(input_sample_poi,hesc_maps, chr, 'hESC')
        HFF_cmap = build_graph(input_sample_poi,HFF_maps, chr, 'HFF')
        contact_maps[chr] = maximum(hesc_cmap, HFF_cmap)
        print(contact_maps[chr].shape[0], contact_maps[chr].nnz)
    return contact_maps

def find_adjs(array):
    nums=array.nnz
    idxs=np.argsort(array.data)[::-1]
    sort_idx=array.indices[idxs]
    if nums>=50:
        idx=sort_idx[:50]
        temp=0
    else:
        idx=sort_idx[:nums]
        temp=50-nums
    return idx.tolist(),temp

def generate_neighbors(input_sample_poi):
    adjs=generate_contact_map(input_sample_poi)
    neighs = {}
    for chr in adjs.keys():
        adj_matrix = csr_matrix(adjs[chr])
        len_adj = adj_matrix.shape[0]
        neigh_list=[len_adj]*5+[n for n in range(len_adj)]+[len_adj]*5
        temp_list = []
        for i in range(len_adj):
            idx, adds = find_adjs(adj_matrix[i, :])
            temp = idx + adds * [len_adj]
            temp.extend(neigh_list[i:i+11])
            temp_list.append(temp)
        neighs[chr] = np.array(temp_list)
    return neighs

import matplotlib
import matplotlib.pyplot as plt

def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
            [0.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.2, 0.0],
        ]),
        np.array([
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.8, 0.0],
        ]),
        np.array([
            [0.225, 0.45],
            [0.775, 0.45],
            [0.85, 0.3],
            [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1, height])[None, :] * polygon_coords
                                                 + np.array([left_edge, base])[None, :]),
                                                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(
        matplotlib.patches.Ellipse(xy=[left_edge + 0.65, base + 0.5 * height], width=0.7 * 1.3, height=0.7 * height,
                                   facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 1, base], width=1.0, height=height,
                                              facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.825, base + 0.085 * height], width=0.174, height=0.415 * height,
                                     facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(
        matplotlib.patches.Rectangle(xy=[left_edge + 0.625, base + 0.35 * height], width=0.374, height=0.15 * height,
                                     facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge + 0.4, base],
                                              width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base + 0.8 * height],
                                              width=1.0, height=0.2 * height, facecolor=color, edgecolor=color,
                                              fill=True))


default_colors = {0: 'green', 1: 'blue', 2: 'orange', 3: 'red'}
default_plot_funcs = {0: plot_a, 1: plot_c, 2: plot_g, 3: plot_t}


def plot_weights_given_ax(tick,ax, array,
                          figsize=(20, 2),
                          height_padding_factor=0.2,
                          length_padding=1.0,
                          subticks_frequency=1.0,
                          colors=default_colors,
                          plot_funcs=default_plot_funcs,
                          highlight={},
                          ylabel=""):
    if len(array.shape) == 3:
        array = np.squeeze(array)
    assert len(array.shape) == 2, array.shape
    if (array.shape[0] == 4 and array.shape[1] != 4):
        array = array.transpose(1, 0)
    assert array.shape[1] == 4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        # sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i, :]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color = colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    # now highlight any desired positions; the key of
    # the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos, min_depth],
                                             width=end_pos - start_pos,
                                             height=max_height - min_depth,
                                             edgecolor=color, fill=False))

    ax.set_xlim(-length_padding, array.shape[0] + length_padding)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0] + 1, 25))
    ax.set_xticklabels(np.arange(tick-400, tick+600, 25), rotation=30)
    ax.yaxis.set_ticks([])
    ax.tick_params(axis='x', labelsize=30,length=30,width=3)
    height_padding = max(abs(min_neg_height) * (height_padding_factor),
                         abs(max_pos_height) * (height_padding_factor))
    ax.set_ylim(min_neg_height - height_padding, max_pos_height + height_padding)
    ax.spines['left'].set_visible(False)
    ax.set_title('Attribution scores on the sequence '+str(tick)+'-'+
                str(tick+200)+' bp with the flanking region ' +
                '\naccording to TF binding on the central sequence',fontsize=80)


def plot_weights(array,tick,
                 figsize=(150, 10),
                 **kwargs):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plot_weights_given_ax(tick,ax=ax, array=array, **kwargs)
    plt.tight_layout()



def plot_score_track_given_ax(arr, ax, threshold=None, **kwargs):
    ax.plot(np.arange(len(arr)), arr, **kwargs)
    if (threshold is not None):
        ax.plot([0, len(arr) - 1], [threshold, threshold])
    ax.set_xlim(0, len(arr) - 1)


def plot_score_track(arr, threshold=None, figsize=(20, 2), **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plot_score_track_given_ax(arr, threshold=threshold, ax=ax, **kwargs)
    plt.show()