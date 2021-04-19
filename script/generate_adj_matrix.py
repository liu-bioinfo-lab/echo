import pickle
import numpy as np
from scipy.sparse import csr_matrix,load_npz,coo_matrix
import scipy.sparse
contact_path='/nfs/turbo/umms-drjieliu/proj/4dn/data/reference_genome_hg38/microc_200bp/'
with open('input_sample_poi.pickle','rb') as f:
    seq_poi=pickle.load(f)
signal_dic={}

def process_contact_map(file,cell_line):
    micro_contact_map={}
    for chr in range(1, 23):
        print(chr)
        micro_contact_map[chr] = {}
        # path = hff_path + 'chr%s_200bp.txt' % chr
        with open(file, 'r') as f:
            for line in f:
                contents = line.strip().split(' ')
                bin1, bin2, value = int(contents[0]), int(contents[1]), int(contents[2])
                if bin1 not in micro_contact_map[chr].keys():
                    micro_contact_map[chr][bin1] = []
                micro_contact_map[chr][bin1].append([bin2, value])
        with open(contact_path + '%s_chr%s.pickle' %(cell_line,chr), 'wb') as f:
            pickle.dump(micro_contact_map, f)
# ratio=[1.885, 1.952, 1.961, 1.931, 1.914, 1.949, 1.783, 1.876, 1.772, 1.87,
#        1.811, 1.909, 1.863, 1.762, 1.9, 1.633, 1.666, 1.813, 1.512, 1.689, 1.682, 1.384]
def generate_contact(cell_line,chr):
    with open(contact_path+'%s_chr%s.pickle'%(cell_line,chr),'rb') as f:
        map=pickle.load(f)
    print('load finished')
    lens=seq_poi[chr].shape[0]
    row=[]
    col=[]
    vals=[]
    for bin in signal_dic[chr].keys():
        try:
            bin_idx1=signal_dic[chr][bin]
            for idx in map[chr][bin]:
                try:
                    bin2=idx[0]
                    if bin== bin2:
                        continue
                    value=idx[1]
                    # value = np.round(idx[1] / ratio[chr - 1], 2)
                    bin_idx2=signal_dic[chr][bin2]
                    row.extend([bin_idx1,bin_idx2])
                    col.extend([bin_idx2,bin_idx1])
                    vals.extend([value,value])
                except Exception:
                    pass
        except Exception:
            pass
    cmap=csr_matrix((np.array(vals),(np.array(row),np.array(col))),shape=(lens,lens))
    scipy.sparse.save_npz('%s_chr%s.npz'%(cell_line,chr),cmap)
def maximum (A, B):
    BisBigger = A-B
    BisBigger.data = np.where(BisBigger.data < 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)
def merge_contact_maps():
    adjs = {}
    for chr in range(1,23):
        print(chr)
        HFF_adj=load_npz('adj_matrix/HFF_chr%s.npz'%chr)
        hESC_adj = load_npz('adj_matrix/hESC_chr%s.npz'%chr)
        adjs[chr]=maximum(HFF_adj,hESC_adj)
    with open('adj_matrix/adjacency_matrix.pickle','wb') as f:
        pickle.dump(adjs,f)
def filter_adjmatrix():
    with open('adj_matrix/adjacency_matrix.pickle','rb') as f:
        adjs=pickle.load(f)
    top_matrix={}
    for chr in range(1,23):
        print(chr)
        matrix=coo_matrix(adjs[chr])
        num=matrix.shape[0]
        row=matrix.row
        col=matrix.col
        data=matrix.data
        idx=np.where(data>=2)[0]
        new_row=row[idx]
        new_col=col[idx]
        new_data=data[idx]
        temp=coo_matrix((new_data,(new_row,new_col)),shape=(num,num))
        top_matrix[chr]=temp
        print(num)
        print(top_matrix[chr].nnz)
    with open('adj_matrix/top_adjacency_matrix_2.pickle','wb') as f:
        pickle.dump(top_matrix,f)