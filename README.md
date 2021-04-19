# ECHO


ECHO (Epigenomic feature analyzer with 3D CHromatin Organization), a graph neural network based model to predict the chromatin features and characterize the collaboration among them in 3D chromatin organization. 

## Overview

## Dependencies

*  python==3.8.5
*  torch==1.7.1
*  scikit-learn==0.23.2
*  numpy==1.19.2
*  scipy==1.5.2


## Usage
### Pre-training the sequence layers
```bash
python pre_train.py
```
### Extracting hidden representations using pre-trained sequence layers
```bash
python hidden_extract.py
```

### Training or testing the graph models
```bash
python graph_train.py
```
