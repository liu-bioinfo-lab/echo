import pickle
path='/nfs/turbo/umms-drjieliu/proj/4dn/data/reference_genome_hg38/'
chroms={}
for chr in range(1,23):
    with open(path+'chr%s.fa'%chr,'r') as f:
        chromosome=''
        for line in f:
            if '>' in line:
                continue
            else:
                chromosome+=line.strip()
    length=len(chromosome)
    chroms[chr]={}

    for i in range(length//200):
        start=i*200
        end=(i+1)*200
        text = chromosome[start:end]
        if 'n' in text.lower():
            continue
        chroms[chr][start]=text
    print('chr%s finished' % chr)
with open('ref_genome_200bp.pickle', 'wb') as handle:
    pickle.dump(chroms,handle)