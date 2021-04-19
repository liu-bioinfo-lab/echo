import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stop
import math
from torch.nn.parameter import Parameter
from GCN import GraphConvolution

"""
Full Chromosome  Models
ChromeGCN
ChromeRNN
Input: DNA window features across an entire chromosome
Output: Epigenomic state prediction for all windows
"""

class ChromeGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(ChromeGCN, self).__init__()
        self.GC1 = GraphConvolution(nfeat, nhid, bias=True,init='xavier')
        self.W1 = nn.Linear(nfeat,1)

        self.GC2 = GraphConvolution(nhid, nfeat, bias=True, init='xavier')
        self.W2 = nn.Linear(nfeat,1)
        self.dropout = 0.2
        self.batch_norm = nn.BatchNorm1d(nfeat)
        self.out = nn.Linear(nfeat,nclass)


    def forward(self, x_in, adj):
        x = x_in
        z = self.GC1(x, adj)
        z = torch.tanh(z)
        g = torch.sigmoid(self.W1(z))
        x = (1-g)*x + (g)*z
        # if hasattr(self, 'GC2'):
        x = F.dropout(x, self.dropout, training=self.training)
        z2 = self.GC2(x, adj)
        z2 = torch.tanh(z2)
        g2 = torch.sigmoid(self.W2(z2))
        x = (1-g2)*x + (g2)*z2

        x = F.relu(x)
        x = self.batch_norm(x)
        x = F.dropout(x, self.dropout, training=self.training)
        out = self.out(x)
        return out