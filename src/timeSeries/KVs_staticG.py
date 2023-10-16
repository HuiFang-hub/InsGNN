# -*- coding: utf-8 -*-
# @Time    : 2023/8/15 17:58
# @Function:
import torch.nn as nn
import torch
import torch.optim as optim
from tslearn.clustering import TimeSeriesKMeans
from src.simple_net import Net


class Shapelet_static(nn.Module):
    def __init__(self, cfg,num_class):
        super().__init__()
        self.cfg = cfg
        self.output_size = num_class
        self.device = cfg.device
        self.cluster = TimeSeriesKMeans(n_clusters=cfg.timeseries.K, metric="dtw", max_iter=10)

        self.network = Net(cfg,self.output_size).to(cfg.device)
        self.opt = optim.Adam(self.network.parameters(), lr=cfg.train.lr)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,data, label):
        # 提取shaplets
        self.cluster.fit(data)
        centers = torch.tensor(self.cluster.cluster_centers_[:, :, -1]).to(self.device)
        # 计算dist
        dist = []
        data = data.to(self.device)
        for t in data:
            d = []
            for c in centers:
                M = torch.cosine_similarity(c, t, dim=0).to(self.device)
                d.append(M)
            d = torch.stack(d)
            dist.append(d)
        # 监督学习
        dist= torch.stack(dist)
        label = label.to(self.device)
        f_list = []
        self.network.train()
        y_pred, y_score = [], []
        tmp = label.cpu().numpy().tolist()
        y_true = list(map(int, tmp))
        l_list = 0
        logit_list = []
        iteration = 0
        for sample, label in zip(dist, label):
            logit,shapelets_logit = self.network(sample)
            logit = logit.squeeze()
            y_score += [logit[1].detach().cpu().numpy().tolist()]
            _, predicted_label = torch.max(logit.data, 0)
            y_pred += [predicted_label.cpu().numpy().tolist()]
            # 求loss
            cur_loss = self.criterion(logit, label.long())
            self.opt.zero_grad()
            cur_loss.backward()
            self.opt.step()
            l_list += cur_loss.item()
            logit_list.append(shapelets_logit.data)
            # f_list.append(self.network.layers[-1].weight.T[:, 1].squeeze())
            iteration += 1

        # f_list = torch.stack(f_list)
        logit_list = torch.stack(logit_list)
        return logit_list



def KVs_staticG(cfg,data,label,num_class):
    shapletE = Shapelet_static(cfg,num_class)
    train_feature = shapletE(data,label)
    return train_feature