from os import listdir
from os.path import isfile, join
import re
import pickle
import numpy as np
path='/nfs/turbo/umms-drjieliu/proj/4dn/data/Epigenomic_Data/chromatin_feature_hg38/tf'
files_dir= [f for f in listdir(path)]
print(len(files_dir))
tf_align={}

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
one_hot_dic={'a':0,'c':1,'g':2,'t':3}
def encoding(text):
    text=text.lower()
    encode_text=np.zeros((4,len(text)))
    for idx in range(len(text)):
        encode_text[one_hot_dic[text[idx]],idx]=1
    return encode_text
for f in files_dir:
    if '.bed' in f:
        file=join(path,f)
        with open(file,'r') as fobj:
            # next(fobj)
            for line in fobj:
                contents=line.strip().split('\t')
                try:
                    chr=int(re.findall('(?<=chr)\d+$',contents[0])[0])
                    if chr not in tf_align.keys():
                        tf_align[chr] = set()
                    start = int(contents[1])
                    end = int(contents[2])
                    for poi in find_locs(start, end):
                        tf_align[chr].add(poi)
                except Exception:
                    pass
ref_gen_path='/nfs/turbo/umms-drjieliu/proj/4dn/data/reference_genome_hg38/ref_genome_200bp.pickle'
with open(ref_gen_path,'rb') as f:
    ref_genome=pickle.load(f)
align_ref={}
for chr in range(1,23):
    align_ref[chr]=[]
    print(chr)
    input_sequence=[]
    temps=np.sort(list(tf_align[chr]))
    for idx in temps:
        try:
            temp = []
            for i in range(-2, 3):
                temp.append(encoding(ref_genome[chr][idx+200*i]))
            temp1 = np.hstack([i for i in temp])
            input_sequence.append(temp1)
            align_ref[chr].append(idx)
        except Exception:
            pass
    np.save('inputs/chr%s.npy' % chr, np.array(input_sequence, dtype=np.int8))
with open('input_sample_poi.pickle','wb') as f:
    pickle.dump(align_ref,f)