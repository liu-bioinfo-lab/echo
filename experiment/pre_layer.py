import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stop
import math
from torch.nn.parameter import Parameter

class Expecto(nn.Module):

    def __init__(self, nclass, seq_length):
        super(Expecto, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4
        sequence_length = seq_length
        n_targets = nclass
        linear_size = 128
        self.src_word_emb = nn.Embedding(5, 5)

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(320, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(320),
            # nn.Dropout(p=0.2),
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
        # x = self.src_word_emb(x)
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 960 * self._n_channels)
        x_feat = self.linear(reshape_out)
        predict = self.relu(x_feat)
        predict = self.batch_norm(predict)
        predict = self.classifier(predict)
        return  predict, x_feat