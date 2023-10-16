# -*- coding: utf-8 -*-
# @Time    : 2023/8/15 18:02
# @Function:
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import InstanceNorm

class ResBlock2(nn.Module):
    def __init__(self,in_channel,middle_channle,out_channel,stride=1):
        super(ResBlock2,self).__init__()

        self.layer=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=middle_channle),
            # nn.BatchNorm2d(middle_channle),
            nn.LeakyReLU(),
            nn.Linear(in_features=middle_channle,out_features=middle_channle),
            # nn.BatchNorm2d(middle_channle),
            nn.LeakyReLU(),
            nn.Linear(in_features=middle_channle,out_features=out_channel),
            # nn.BatchNorm2d(out_channel)
        )
        self.shortcut=nn.Sequential()
        if in_channel != out_channel or stride>1:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features=in_channel,out_features=out_channel),
                # nn.BatchNorm2d(out_channel)
            )
        self.initialize()

    def forward(self,x):
        output1=self.layer(x)
        output2=self.shortcut(x)
        output3=output1+output2
        output=F.relu(output3)
        return output

    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                # m.bias.data.fill_(2)


class Net(nn.Module):
    def __init__(self,cfg,output_size):
        super(Net,self).__init__()
        hidden_dim = cfg.model.hidden_size
        self.network = nn.ModuleList()
        net = nn.Linear(cfg.timeseries.K, hidden_dim)
        nn.init.xavier_normal_(net.weight)
        self.network.append(net)
        self.network.append(nn.LeakyReLU())
        for _ in range(cfg.model.num_mlp_layers):
            net = ResBlock2(hidden_dim, 16,hidden_dim)
            self.network.append(net)
        self.relu = nn.ReLU()
        net = nn.Linear(hidden_dim, cfg.timeseries.shaplet_outsize)
        nn.init.xavier_normal_(net.weight)
        self.network.append(net)
        self.network.append(nn.LeakyReLU())
        net = nn.Linear( cfg.timeseries.shaplet_outsize,output_size)
        nn.init.xavier_normal_(net.weight)
        self.network.append(net)

    def forward(self,input):
        layer_input = input.to(torch.float32)
        shapelets_logit = None
        for layer_id in range(len(self.network)):
            if layer_id == len(self.network)-2:
                shapelets_logit = self.network[layer_id](layer_input)
            layer_input = self.network[layer_id](layer_input)
        return layer_input,shapelets_logit

class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs

class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)
