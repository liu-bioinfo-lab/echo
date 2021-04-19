import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECHO(nn.Module):
    def __init__(self, nclass,k,k_neigh):
        """Dense version of GAT."""
        super(ECHO, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(128,320,1),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Conv1d(320,320, 1),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Conv1d(320, 640, k+1),
            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(640, 960, 1),
            nn.BatchNorm1d(960),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.layers1 = nn.Sequential(
            nn.Conv1d(128, 320, 1),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Conv1d(320, 320, 1),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Conv1d(320, 640, k_neigh + 1),
            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(640, 960, 1),
            nn.BatchNorm1d(960),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.linear=nn.Linear((960+960)*2,nclass)
        # self.linear1=nn.Linear(128,nclass)
    def forward(self, x,x_neigh,x_rev,x_neigh_rev):
        x=x.permute(0,2,1)
        x=torch.squeeze(self.layers(x))
        x_rev = x_rev.permute(0, 2, 1)
        x_rev = torch.squeeze(self.layers(x_rev))

        x_neigh=x_neigh.permute(0,2,1)
        x_neigh=torch.squeeze(self.layers1(x_neigh))

        x_neigh_rev = x_neigh_rev.permute(0, 2, 1)
        x_neigh_rev = torch.squeeze(self.layers1(x_neigh_rev))

        x_fea = torch.cat((x, x_neigh,x_rev,x_neigh_rev), 1)
        x=self.linear(x_fea)
        return x