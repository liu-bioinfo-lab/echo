# ECHO


ECHO (Epigenomic feature analyzer with 3D CHromatin Organization), a graph neural network based model to predict the chromatin features and characterize the collaboration among them in 3D chromatin organization. 


<!-- ## Dependencies

*  python==3.8.5
*  torch==1.7.1
*  scikit-learn==0.23.2
*  numpy==1.19.2
*  scipy==1.5.2 -->

## Methods
### Model architecture

<div align=center><img src="./doc/architecture.png" width="600px"></div>

### Applying attribution methods to ECHO

<div align=center><img src="./doc/attribution.png" width="600px"></div>


## Usage


Please see [neighborhood_motif.ipynb](neighborhood_motif.ipynb) for an example to find common motif patterns in the nerighborhood for the inverstigated chromatin feature using [TF-MoDISco (Shrikumar et al.)](https://github.com/kundajelab/tfmodisco). Please refer to attribute_central_sequence.py for calculating the attribution scores of the central sequence which can be used to generate binding motifs by using TF-MoDISco.

If you want to train the model from scratch, please refer to the code below,. 

```bash 
python pre_train.py --lr=0.5 --pre_model=expecto --batchsize=64 --length=2600 --seq_length=1000
python hidden_extract.py --pre_model=expecto --length=2600
python graph_train.py --lr=0.5 --batchsize=64 --k_adj=50 --k_neigh=10 --pre_model=expecto
```
<!-- 
### Attribution scores on chromatin contacts according to certain chromatin feature prediction
e.g. attribute GM12878 H3k4me3 prediction to chromatin contacts

```bash
python attribution_contact.py --chromatin_feature=h3k4me3 --cell_line=gm12878
``` -->


## Data

For the collected chromatin feature profiles, please see [chromatin_feature_profiles.xlsx](https://github.com/liu-bioinfo-lab/echo/blob/main/doc/chromatin_feature_profiles.xlsx)

<!-- Please see https://drive.google.com/drive/folders/1rI9WRPb_MwM36sW6AH7INC63Vo5fVelb?usp=sharing for the label data.

Our input sequence data can be generated using the codes below with the downloaded reference genome data 
```bash
from util1 import generate_inputs
import pickle
with open('example/input_sample_poi.pickle','rb') as f:
  input_sample_poi=pickle.load(f)
with open('echo_data/ref_genome_200bp.pickle','rb') as f:
  ref_genome=pickle.load(f)
inputs={}
for chr in range(1,23):
  inputs[chr]=generate_inputs(input_sample_poi,chr,ref_genome)
``` -->

Our neighhood data can be downloaded using the command lines below

```bash
pip install gdown
gdown --id 1nx8pRvG5CWkGQINS_f5Uk451A0Tt4NY5 --output neighbors_data.zip
unzip neighbors_data.zip
```



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
