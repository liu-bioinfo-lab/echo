from os import listdir
from os.path import isfile, join
import re
import pickle,argparse
import numpy as np
from scipy.sparse import csr_matrix,save_npz

def find_locs(start,end):
    sets=[]
    s=start//200
    e=end//200+1 if end%200 else end//200
    if s==e-1:
        if end-start>=100:
            sets.append(s*200)
    elif s==e-2:
        if (s + 1) * 200 - start >=100:
            sets.append(s*200)
        if end-(e-1)*200>=100:
            sets.append((e-1) * 200)
    else:
        if (s + 1) * 200 - start >= 100:
            sets.append(s*200)
        for i in range(s+1,e-1):
            sets.append(i*200)
        if end-(e-1)*200>=100:
            sets.append((e-1) * 200)
    return sets
dir_path='/nfs/turbo/umms-drjieliu/usr/zzh/deepchrom/'
with open(dir_path+'input_sample_poi.pickle','rb') as f:
    seq_poi=pickle.load(f)
def label_bins(feature_type):
    label_dics={}
    label = {}
    if feature_type.lower()=='histone':
        path='/nfs/turbo/umms-drjieliu/proj/4dn/data/Epigenomic_Data/chromatin_feature_hg38/ihec_histone'
    else:
        path = '/nfs/turbo/umms-drjieliu/proj/4dn/data/Epigenomic_Data/chromatin_feature_hg38/%s'%feature_type
    files_dir = [f for f in listdir(path)]
    nums=len(files_dir)
    print(nums)
    for chr in range(1,23):
        label[chr]={}
        for idx in seq_poi[chr]:
            label[chr][idx]=np.zeros(nums)
    for idx in range(len(files_dir)):
        print(files_dir[idx])
        # if '.bed' in files_dir[idx]:
        file = join(path, files_dir[idx])
        with open(file,'r') as fobj:
            for line in fobj:
                contents=line.strip().split('\t')
                chr =re.findall('(?<=chr)\d+$',contents[0])
                if chr:
                    chr=int(chr[0])
                    start = int(contents[1])
                    end = int(contents[2])
                    for poi in find_locs(start, end):
                        try:
                            label[chr][poi][idx]=1
                        except Exception:
                            pass

    for chr in range(1,23):
        templ = []
        for idx in seq_poi[chr]:
            templ.append(label[chr][idx])
        label_dics[chr] =np.array(templ,dtype=np.int8)
    return label_dics
        # np.save('labels/%s/chr%s.npy' %(feature_type,chr), np.array(templ,dtype=np.int8))
chromatin_feature_dic={}
for ctype in ['tf','histone','dnase']:
    chromatin_feature_dic[ctype] =label_bins(ctype)
for chr in range(1,23):
    temp = []
    for ctype in ['tf','histone','dnase']:
        temp.append(chromatin_feature_dic[ctype][chr])
    output_label = np.hstack([i for i in temp])
    output_label = csr_matrix(output_label)
    save_npz(dir_path+'labels/ihec_labels/chr%s.npz' % chr, output_label)

