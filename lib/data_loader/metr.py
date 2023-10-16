# -*- coding: utf-8 -*-
# @Time    : 2023/9/11 18:31
# @Function:
import os
import re
import pickle
import pandas as pd
from lib.data_loader.process_data import data_normal_process
from torch_geometric.data import Data, InMemoryDataset
import torch
from glob2 import glob
from mne.io import RawArray, read_raw_edf
import dgl
import numpy as np
from collections import Counter
import scipy.sparse as sp
from src.timeSeries.KVs_dyG import  kvs
from torch_geometric.utils import from_scipy_sparse_matrix
from chinese_calendar import is_holiday

class metrDataset(InMemoryDataset):
    def __init__(self,cfg, root,transform=None, pre_transform=None):
        self.cfg = cfg
        self.length = cfg.data.ts_length
        self.l_min = cfg.data.l_min
        self.shapelet = True
        # self.channels_name = ['FP1-F7', 'P7-O1', 'F3-C3', 'C3-P3', 'FP2-F8', 'P8-O2', 'F4-C4', 'C4-P4', 'T7-FT9', 'FT10-T8']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.shapelet:
            return ['shapelet_data.pt']
        else:
            return ['data.pt']

    @property
    def _download(self):
        return

    def process(self):
        edge_index = self.edge_load()

        data_list = []
        ts_df = pd.read_hdf(os.path.join(self.raw_dir,'metr-la.h5'))
        ts_df.index = pd.to_datetime(ts_df.index)
        self.train_f_channels_path = os.path.join(self.processed_dir, 'train_f_channels')
        self.centroids_path = os.path.join(self.processed_dir, 'centroids')
        if not os.path.exists(self.train_f_channels_path):
            os.makedirs(self.train_f_channels_path)
        if  not os.path.exists(self.centroids_path):
            os.makedirs(self.centroids_path)
        labels = []
        if self.shapelet:
            dates = ts_df.index.date
            unique_dates = sorted(set(dates))
            for date in unique_dates:
                y= torch.tensor(is_holiday(date), dtype=torch.long)
                labels.append(y)
            # labels = torch.tensor(labels)
            for node_name, node_fea in ts_df.iteritems():
                train_f_channel_path = os.path.join(self.train_f_channels_path, f'{node_name}.pkl')
                if os.path.exists(train_f_channel_path):
                    continue
                else:
                    fea = []
                    for day, day_df in node_fea.groupby(pd.Grouper(freq='D')):
                        # y = is_holiday(day)
                        # labels.append(float(y))
                        fea.append(day_df.tolist())

                    feature = data_normal_process(fea, self.cfg)
                    pickle.dump(feature, open(train_f_channel_path, 'wb'))
        else:
            for day, day_df in ts_df.groupby(pd.Grouper(freq='D')):
                y = torch.tensor(is_holiday(day)).float().reshape(-1, 1)
                x = torch.from_numpy(day_df.values.T)
                # dgl
                u, v = edge_index[0], edge_index[1]
                g = dgl.graph((u, v))
                x = x[:g.num_nodes()]
                g.ndata['feat'] = x.float()
                data_list.append(Data(graph=g, x=x, y=y, edge_index=edge_index))



        if self.shapelet:
            # feas = KVs_dyG(self.cfg, self.train_f_channels_path, labels, self.processed_dir)
            # test = ts_df.columns.tolist()
            compress_fea_path =  os.path.join(self.processed_dir, 'compressed_fea.pkl')
            if not os.path.exists(compress_fea_path):
                compressed_fea = self.fea_to_KVs(labels, ts_df.columns.tolist())
                pickle.dump(compressed_fea, open(compress_fea_path, 'wb'))
            else:
                with open(compress_fea_path , "rb") as f:
                    compressed_fea =  pickle.load(f)

            for label in  labels:
                y = torch.tensor(label).reshape(-1, 1)
                x = compressed_fea[:2].T
                compressed_fea = compressed_fea[2:]
                u, v = edge_index[0], edge_index[1]
                g = dgl.graph((u, v))
                # x = x[:g.num_nodes()]
                g.ndata['feat'] = x.float()
                data_list.append(Data(graph=g, x=x, y=y, edge_index=edge_index))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # if index== index2:
        #     print("yes")
        # print('t')


    def edge_load(self):
        with open(os.path.join(self.raw_dir,'adj_mat.pkl'), "rb") as f:
            adj = pickle.load(f)
        coo_adj =sp.coo_matrix(adj[2])
        edge_index = torch.tensor([ coo_adj.row,  coo_adj.col],dtype = torch.long)
        return edge_index

    def label_load(self,time_list):
        time = pd.to_datetime(time_list)
        labels = time.apply(func=is_holidays)
        return labels.values

    def fea_to_KVs(self,labels,node_names):
        outsize = len(np.bincount(labels))
        kv_model = kvs(self.cfg, self.processed_dir,outsize)
        compressed_fea = torch.tensor([])
        for name in node_names:
            f_path = os.path.join(self.train_f_channels_path,f'{name}.pkl')
            c_path = os.path.join(self.centroids_path,f'{name}.pkl')
            # with open(f_path, "rb") as f:
            #     train_f_channel = pickle.load(f)

            fea = kv_model.train(f_path,c_path,labels)
            compressed_fea = torch.cat((compressed_fea, fea), dim=1)
        return compressed_fea






def is_holidays(x):
    return  int(is_holiday(x))




