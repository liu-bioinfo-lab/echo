# ECHO


ECHO (Epigenomic feature analyzer with 3D CHromatin Organization), a graph neural network based model to predict the chromatin features and characterize the collaboration among them in 3D chromatin organization. 

## Methods
### Model architecture
![Screenshot](./doc/architecture.png)
### Applying attribution methods to ECHO
![Screenshot](./doc/attribution.png)
## Dependencies

*  python==3.8.5
*  torch==1.7.1
*  scikit-learn==0.23.2
*  numpy==1.19.2
*  scipy==1.5.2


## Usage
### Model training
pre-train sequence layers
```bash
python pre_train.py
```
extracting hidden representations using pre-trained sequence layers
```bash
python hidden_extract.py
```
training the graph layers with the extracted sequence hidden representations
```bash
python graph_train.py
```
In ```\models\```, we provide the trained models.
### Calculate attribution scores of Micro-C contact matrix
e.g. attribute CTCF labels to the contact matrix
```bash 
python attribution_contact.py --chromatin_feature= ctcf
```
### Calculate attribution scores for the neighborhood 
e.g. attribute GM12878 H3k4me3 to the neighbor sequences 

First, get the corresponding attributed contact matrix
```bash
python attribution_contact.py --chromatin_feature= h3k4me3 --cell_line=gm12878
```
Next, calculate the attribution scores for selected neighbor sequences, patterns can be learnen from the neighbor sequences by using the tool TF-MoDISco
```bash
python attribution_neighborhood.py --chromatin_feature= h3k4me3 --cell_line=gm12878
```
