import torch
import torch.nn as nn
def cnn_block(in_channels, out_channels, kernel_size,strides, padding):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size,strides, padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        # nn.Dropout(0.2),
        nn.Conv1d(out_channels, out_channels, kernel_size=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        )

class ECHO(nn.Module):
    def __init__(self, nclass,k,k_neigh):
        super(ECHO, self).__init__()
        self.layers=nn.Sequential(
            cnn_block(k,128,20,1,0),
            nn.MaxPool1d(5),
            cnn_block(128,240,10,1,0),
            nn.MaxPool1d(5),
            nn.Dropout(0.2),
            cnn_block(240,320,10,1,0),
            nn.MaxPool1d(5),
            nn.Dropout(0.2),
            cnn_block(320,1300,5,1,0),
            nn.Dropout(0.3)
        )
        self.layers1 = nn.Sequential(
            cnn_block(k_neigh + 1,128,20,1,0),
            nn.MaxPool1d(5),
            cnn_block(128,240,10,1,0),
            nn.MaxPool1d(5),
            nn.Dropout(0.2),
            cnn_block(240,320,10,1,0),
            nn.MaxPool1d(5),
            nn.Dropout(0.2),
            cnn_block(320,1300,5,1,0),
            nn.Dropout(0.3)
        )
        self.linear = nn.Linear(1300+1300, nclass)
    def forward(self, x,x_neigh):
        x= self.layers(x)
        x=x.mean(2)
        x1=self.layers1(x_neigh)
        x1=x1.mean(2)
        x_fea=torch.cat((x,x1),1)
        x=self.linear(x_fea)
        return x
