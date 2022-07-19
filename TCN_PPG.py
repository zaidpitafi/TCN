import torch.nn as nn
from torch.nn.utils import weight_norm
import torch
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.1):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size-1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                            stride=1, padding=padding, dilation=dilation))
        
        self.chomp1 = Chomp1d(padding) 
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                            stride=1, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
    

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        return out

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_fusion = len(kernel_size)
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            
            in_channels = num_inputs if i == 0 else num_channels[i-1]  
            out_channels = num_channels[i]  
            for j in range(len(kernel_size)):
                self.layers += [TemporalBlock(in_channels, out_channels, kernel_size[j], dilation=dilation_size, dropout=dropout)]
        self.average_pooling = nn.AvgPool1d(64)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 4)
        self.fc2 = nn.Linear(64, 4)
        self.downsample = nn.Conv1d(num_inputs, num_channels[0], 1)
        self.conv = nn.Conv1d(out_channels, 1, kernel_size = 1)
        self.init_weights()

    def init_weights(self):
        self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        i = 0
        while(i < len(self.layers)):
            for j in range(self.n_fusion):
                if j == 0:
                    features = self.layers[i + j](x)
                else:
                    features = self.layers[i + j](x) + features

            if (x.shape[1] != features.shape[1]):
                x = self.downsample(x)

            x = nn.ReLU()(features + x)
            i = i + self.n_fusion
        x = self.conv(x)
        # x = x.permute(0,2,1)
        # x = self.average_pooling(x)
        # x = x.permute(0,2,1)
        x = self.flatten(x)
        
        #######################################################
        x_sp = self.dropout(x)
        x_sp = self.fc1(x)
        x_dp = self.dropout2(x)
        x_dp = self.fc2(x)
        sp_v, sp_mean, sp_alpha, sp_beta = torch.split(x_sp, split_size_or_sections = 1, dim = 1)
        dp_v, dp_mean, dp_alpha, dp_beta = torch.split(x_dp, split_size_or_sections = 1, dim = 1)
        #######################################################
        
        sp_v = nn.Softplus()(sp_v)
        sp_alpha = nn.Softplus()(sp_alpha) + 1
        sp_beta = nn.Softplus()(sp_beta)
        dp_v = nn.Softplus()(dp_v)
        dp_alpha = nn.Softplus()(dp_alpha) + 1
        dp_beta = nn.Softplus()(dp_beta)
        sp = torch.cat([sp_v, sp_mean, sp_alpha, sp_beta], axis= 1)
        dp = torch.cat([dp_v, dp_mean, dp_alpha, dp_beta], axis= 1)
        return sp, dp
