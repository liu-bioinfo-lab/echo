import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
class Expecto(nn.Module):
    def __init__(self, nclass, seq_length,embed_length):
        super(Expecto, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4
        sequence_length = seq_length
        n_targets = nclass
        linear_size = embed_length
        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(320, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(320),
            nn.Dropout(p=0.2),
            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(480),
            nn.Dropout(p=0.2),
            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.5))
        reduce_by = 2 * (conv_kernel_size - 1)
        pool_kernel_size = float(pool_kernel_size)
        self._n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        self.linear = nn.Linear(960 * self._n_channels, linear_size)
        self.batch_norm = nn.BatchNorm1d(linear_size)
        self.classifier = nn.Linear(linear_size, n_targets)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 960 * self._n_channels)
        x_feat = self.linear(reshape_out)
        predict = self.relu(x_feat)
        predict = self.batch_norm(predict)
        predict = self.classifier(predict)
        return  predict, x_feat

class DeepSEA(nn.Module):
    def __init__(self,nclass,seq_length,embed_length):
        super(DeepSEA, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4
        sequence_length = seq_length
        n_targets = nclass
        linear_size = embed_length
        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),
            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),
            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))
        reduce_by = conv_kernel_size - 1
        pool_kernel_size = float(pool_kernel_size)
        self.n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by) / pool_kernel_size)
                 - reduce_by) / pool_kernel_size)
            - reduce_by)
        self.linear = nn.Linear(960 * self.n_channels, linear_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(linear_size, n_targets)
    def forward(self, x):
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 960 * self.n_channels)
        xfea = self.linear(reshape_out)
        x = self.relu(xfea)
        predict = self.classifier(x)
        return predict,xfea

class DanQ(nn.Module):
    def __init__(self, nclass,seq_length,embed_length):
        super(DanQ, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26)
        self.Maxpool = nn.MaxPool1d(kernel_size=13, stride=13)
        self.Drop1 = nn.Dropout(p=0.2)
        self.BiLSTM = nn.LSTM(input_size=320, hidden_size=320, num_layers=2,
                                 batch_first=True,
                                 dropout=0.5,
                                 bidirectional=True)
        self.channel=int(np.floor((seq_length-25)/13))
        self.Linear1 = nn.Linear(640*self.channel, embed_length)
        self.Linear2 = nn.Linear(embed_length, nclass)

    def forward(self, x):
        x = self.Conv1(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x_x = torch.transpose(x, 1, 2)
        x, (h_n,h_c) = self.BiLSTM(x_x)
        x = x.contiguous().view(-1, 640*self.channel)
        x = self.Linear1(x)
        x_feat = F.relu(x)
        predict = self.Linear2(F.relu(x))
        return predict,x_feat
