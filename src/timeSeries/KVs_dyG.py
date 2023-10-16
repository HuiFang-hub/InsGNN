# -*- coding: utf-8 -*-
# @Time    : 2023/8/15 17:58
# @Function:
import os
import pickle

import torch.nn as nn
import torch
import torch.optim as optim
from tslearn.clustering import TimeSeriesKMeans
from src.simple_net import Net
from src.timeSeries.utils import Network
from src.simple_net import ResBlock2
from src.timeSeries.utils import get_centroids_of_segments
from src.timeSeries.soft_min_layer import SoftMinLayer
from src.timeSeries.aggregation_layer import AggregationLayer
import torch.nn.functional as F
from tqdm import tqdm
# class Shapelet_staticG4(nn.Module):
#     def __init__(self, cfg,num_class,multi_label,l_min):
#         super().__init__()
#         self.cfg = cfg
#         self.output_size = num_class
#         self.device = cfg.device
#         self.cluster = TimeSeriesKMeans(n_clusters=cfg.K, metric="dtw", max_iter=10)
#         self.hidden_dim = cfg.hidden_dim
# 
#         self.network = Net(cfg,self.hidden_dim,self.output_size).to(cfg.device)
#         self.opt = optim.Adam(self.network.parameters(), lr=cfg.lr)
#         self.criterion = nn.CrossEntropyLoss()
# 
#     def forward(self,data, label):
#         # 提取shaplets
#         self.cluster.fit(data)
#         centers = torch.tensor(self.cluster.cluster_centers_[:, :, -1]).to(self.device)
#         # 计算dist
#         dist = []
#         data = data.to(self.device)
#         for t in data:
#             d = []
#             for c in centers:
#                 M = torch.cosine_similarity(c, t, dim=0).to(self.device)
#                 d.append(M)
#             d = torch.stack(d)
#             dist.append(d)
#         # 监督学习
#         dist= torch.stack(dist)
#         label = label.to(self.device)
#         f_list = []
#         self.network.train()
#         y_pred, y_score = [], []
#         tmp = label.cpu().numpy().tolist()
#         y_true = list(map(int, tmp))
#         l_list = 0
#         logit_list = []
#         iteration = 0
#         for sample, label in zip(dist, label):
#             logit = self.network(sample)
#             logit = logit.squeeze()
#             y_score += [logit[1].detach().cpu().numpy().tolist()]
#             _, predicted_label = torch.max(logit.data, 0)
#             y_pred += [predicted_label.cpu().numpy().tolist()]
#             # 求loss
#             cur_loss = self.criterion(logit, label.long())
#             self.opt.zero_grad()
#             cur_loss.backward()
#             self.opt.step()
#             l_list += cur_loss.item()
#             logit_list.append(logit.data)
#             # f_list.append(self.network.layers[-1].weight.T[:, 1].squeeze())
#             iteration += 1
# 
#         # f_list = torch.stack(f_list)
#         logit_list = torch.stack(logit_list)
#         return logit_list



class kvs(nn.Module):
    def __init__(self,cfg,processed_dir,outsize):
        super(kvs, self).__init__()
        self.cfg = cfg
        self.alpha = cfg.timeseries.alpha
        self.eta = cfg.timeseries.eta
        self.lamda = cfg.timeseries.lamda
        # self.train_size = train_size
        # self.networkDic = {}
        self.K = cfg.timeseries.K
        self.n_shapelets = cfg.timeseries.R
        self.R = cfg.timeseries.R
        self.L_min= cfg.data.l_min
        self.shapelet_initialization = cfg.timeseries.shaplet_method
        self.device = cfg.device
        self.lr = cfg.train.lr
        self.shaplet_outsize = cfg.timeseries.shaplet_outsize
        # self.train_f_channels = {}
        self.folder_path = processed_dir
        self.outsize = outsize

        # self.train_f_channels_path = train_f_channels_path
        # self.top_K_centroids_path = top_K_centroids_path
        # if os.path.exists(train_f_channels_path):
        #     with open(train_f_channels_path, "rb") as f:
        #         self.train_f_channels = pickle.load(f)
        # else:
        #     for channel in self.channels_name:
        #         print('Using channel data %s to initialize shaplets'%(channel))
        #         train_f_channel = ShapeletsData_process(train_data,channel,cfg)
        #         train_f_channel = TimeSeriesScalerMinMax().fit_transform(train_f_channel)[:, :, -1]
        #         train_f_channel = torch.from_numpy(train_f_channel).to(cfg.device)
        #         self.train_f_channels[channel] = train_f_channel
        #         pickle.dump(self.train_f_channels, open(train_f_channels_path, 'wb'))
        # self.parameters = []
        # for channel in self.channels_name:
        #     # self.train_f_channel = self.train_f_channels[channel]
        #     shapletlayer = self._init_network_shapelets(channel).to(self.device)
        #     self.networkDic[channel] = shapletlayer
        #     self.parameters += [{'params': self.networkDic[channel].parameters(), 'lr': self.lr}]


    def _init_network_shapelets(self):
        self.networks = Network()
        # shapelets layer
        self.networks.add_layer(self._create_shapelets_layer_segments_centroids())
        # neural network
        net = nn.Linear(self.n_shapelets, 128)
        nn.init.xavier_normal_(net.weight)
        self.networks.add_layer(net)
        self.networks.add_layer(nn.LeakyReLU())
        # # resnet
        for _ in range(2):
            self.networks.add_layer(ResBlock2(128, 64, 128))
        # output layer
        net = nn.Linear(128, self.shaplet_outsize )
        nn.init.xavier_normal_(net.weight)
        # nn.init.xavier_normal_(net.bias)
        self.networks.add_layer(net)
        self.networks.add_layer(nn.LeakyReLU())

        net = nn.Linear(self.shaplet_outsize,self.outsize)
        nn.init.xavier_normal_(net.weight)
        # nn.init.xavier_normal_(net.bias)
        self.networks.add_layer(net)
        self.networks.add_layer(nn.LeakyReLU())


        self.networks = self.networks.to(self.device)
        self.opt = optim.Adam(self.networks.parameters(), lr=self.cfg.train.lr)
        # self.criterion = F.binary_cross_entropy_with_logits() #nn.CrossEntropyLoss()


    def _create_shapelets_layer_segments_centroids(self):
        # Shapelets are included in SoftMinLayers
        min_soft_layers = []
        self.top_K_centroids_scale_r = []
        # cnt=0
        if os.path.exists(self.top_K_centroids_path):
            with open(self.top_K_centroids_path, "rb") as f:
                self.top_K_centroids_scale_r = pickle.load(f)
        else:
            for r in range(self.R):
                L = (r+1) * self.L_min
                # 这一步太耗时
                # path = self.folder_path + '/train_process/train_f_channels_%s.pkl' % self.channel_name
                with open(self.train_f_channel_path, "rb") as f:
                    train_f_channel = pickle.load(f)
                self.top_K_centroids_scale_r.append(get_centroids_of_segments(train_f_channel, L, self.K))

            pickle.dump(self.top_K_centroids_scale_r, open(self.top_K_centroids_path, 'wb'))
            # 这一步也耗时
        for centroid in self.top_K_centroids_scale_r:
            # centroid = np.array([centroid])
            centroid = torch.from_numpy(centroid).to(self.device)
            min_soft_layers.append(
                SoftMinLayer(self.cfg, centroid, self.eta, self.alpha))
        # cnt += 1
        # shapelets aggregation layer
        aggregator = AggregationLayer(min_soft_layers, self.device)
        return aggregator

    def forward(self,X):
        current_output = []
        for channel in self.channels_name:
            out = self.networkDic[channel].forward(X)
            current_output.append(out)
        return torch.cat(current_output, dim=1)

    def backword(self):
        for channel in self.channels_name:
            self.networkDic[channel].backward()
        
    def train(self,f_path,c_path,labels):
        self.train_f_channel_path= f_path
        self.top_K_centroids_path = c_path
        self. _init_network_shapelets()
        self.networks.train()
        logit = None
        with open(self.train_f_channel_path, "rb") as f:
            data = pickle.load(f)
        # 创建一个tqdm进度条
        for epoch in tqdm(range(self.cfg.timeseries.epochs), desc='Training Progress'):
            for x, label in zip(data, labels):
                x = torch.tensor(x, device=self.device)
                label = label.to(self.device).unsqueeze(0)
                logit = self.networks.forward(x)
                loss = F.cross_entropy(logit, label)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        self.networks.eval()
        logits = []
        for x, label in tqdm(zip(data, labels), desc='Processing Data'):
            x = torch.tensor(x, device=self.device)
            logit = self.networks.forward(x)
            logit = logit.squeeze().detach().cpu()
            logits.append(logit)

        logits = torch.cat(logits).unsqueeze(1) # (-1,1)
        return logits
        
#
# def KVs_dyG(cfg,channel_fea_dict,labels,processed_dir):
#     train_features = []
#     for (channel,f_channel),label in zip(channel_fea_dict.items(),labels):
#         shapletE = kvs(cfg, channel, processed_dir)
#         train_feature = shapletE.train(f_channel,label).reshape(-1,1)
#         train_features.extend(train_feature)
    # return train_features