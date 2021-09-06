# ECHO


ECHO (Epigenomic feature analyzer with 3D CHromatin Organization), a graph neural network based model to predict the chromatin features and characterize the collaboration among them in 3D chromatin organization. 

## Methods
### Model architecture

<div align=center><img src="./doc/architecture.png" width="600px"></div>

### Applying attribution methods to ECHO

<div align=center><img src="./doc/attribution.png" width="600px"></div>

<!-- ## Dependencies

*  python==3.8.5
*  torch==1.7.1
*  scikit-learn==0.23.2
*  numpy==1.19.2
*  scipy==1.5.2 -->

## Usage

Please see [echo_example.ipynb](echo_example.ipynb) for an example to calulate prediction scores for the central sequence and attribution scores on chromatin contacts and neighbor sequences.

To perform your own analysis, you only need to provide an ATAC-seq file in (.bed) format. The chromatin contacts data will be provided by our 200-bp merged HFF and hESC Micro-C contact maps.

First, you need to download the Micro-C contact maps and GRCh38 reference genome data
```bash
curl -L -o reference_genome.zip "https://drive.google.com/uc?export=download&id=11NXyB2FNehMRw_lbCe7_Ymlp86w5TQdR"
unzip reference_genome.zip
curl -L -o HFF_Micro-C.zip "https://drive.google.com/uc?export=download&id=1XAGqlZsZJ6CY2wQMVVvNXnlnVL9iEmtf"
unzip HFF_Micro-C.zip
curl -L -o hESC_Micro-C.zip "https://drive.google.com/uc?export=download&id=1aFK1wadoZ1h0Y0f6UTp76XWcbN5xqGe3"
unzip hESC_Micro-C.zip
```

Then, you can generate the input data ('inputs','neighs','input_sample_poi'). p.s. if your uploaded ATAC-seq data is GRCh37(hg19), it will be converted to GRCh38(hg38).
```bash
from util import *
inputs,input_sample_poi=generate_input(ATAC_seq_file,version='hg38')
neighs= generate_neighbors(input_sample_poi)
```

To run the analysis, you can follow the steps demonstrated in [echo_example.ipynb](echo_example.ipynb).

<!-- ## Usage
In ```\utils\```, we provide the code for pre-processing data
### Model training
pre-train sequence layers 
```bash
python pre_train.py --lr=0.5 --pre_model=expecto --batchsize=64 --length=2600 --seq_length=1000
```
extracting hidden representations using pre-trained sequence layers 
```bash
python hidden_extract.py --pre_model=expecto --length=2600
```
training the graph layers with the extracted sequence hidden representations
```bash
python graph_train.py --lr=0.5 --batchsize=64 --k_adj=50 --k_neigh=10 --pre_model=expecto
```
Add ```--load_model``` for loading trained models, add ```--test``` for model testing.

In ```\models\```, we provide the trained models.
### Calculate attribution scores of Micro-C contact matrix
For the collected chromatin features profiles, please check  ```\doc\chromatin_feature_profiles.xlsx```
```bash 
python attribution_contact.py --chromatin_feature=ctcf --k_adj=50 --k_neigh=10
```
### Calculate attribution scores for the neighborhood 
e.g. attribute GM12878 H3k4me3 to the neighbor sequences 

First, get the corresponding attributed contact matrix
```bash
python attribution_contact.py --chromatin_feature=h3k4me3 --cell_line=gm12878
```
Next, calculate the attribution scores for selected neighbor sequences, patterns can be learnen from the neighbor sequences by using the tool TF-MoDISco
```bash
python attribution_neighborhood.py --chromatin_feature=h3k4me3 --cell_line=gm12878
```
 -->
