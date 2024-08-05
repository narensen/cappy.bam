import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as  F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_rate):
        super(ChannelAttention, self).__init__()
        self.inchannels = in_channels
        self.reduction = reduction_rate
        self.hn = in_channels // reduction_rate

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.batch_norm = nn.BatchNorm1d(in_channels)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, self.hn),
            nn.ReLU(),
            nn.Linear(self.hn, in_channels),
            nn.Sigmoid())
        
    def forward(self,x):

        gap = self.avg_pool(x)
        gap = gap.view(gap.size(0), -1)
        gap = self.batch_norm(gap)

        attention_weights = self.mlp(gap)
        attention_weights = attention_weights.view(attention_weights.size(0), self.inchannels, 1, 1)

        out = x * attention_weights

        return out
    
class SpatialAttention(nn.Module):

    def __init__(self, in_channels, reduction_rate):
        super(SpatialAttention, self).__init__()

        self.inchannels = in_channels
        self.reduction = reduction_rate

        hn = in_channels // reduction_rate

        self.conv1x1 = nn.Conv2d(in_channels, hn, kernel_size=1)
        self.batch_norm1x1 = nn.BatchNorm2d(hn)

        self.conv3x3 = nn.Conv2d(hn, hn, kernel_size=3, dilation=4)
        self.batch_norm3x3 = nn.BatchNorm2d(hn)

        self.conv3x3_ = nn.Conv2d(hn, hn, kernel_size=3, dilation=4)
        self.batch_norm3x3_ = nn.BatchNorm2d(hn)

        self.conv_ = nn.Conv2d(hn, 1, kernel_size=1)
        self.batch_norm_fin = nn.BatchNorm2d(hn)

    def forward(self, x):

        x = self.conv1x1(x)
        x = self.batch_norm1x1(x)
        x = F.relu(x)

        x = self.conv3x3(x)
        x = self.batch_norm3x3(x)
        x = F.relu(x)

        x = self.conv3x3_(x)
        x = self.batch_norm3x3_
        x = F.relu(x)

        x = self.conv_(x)
        x = self.batch_norm_fin(x)
        x = F.relu(x)

        return x
    

    
    



        