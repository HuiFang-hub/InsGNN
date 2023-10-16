# -*- coding: utf-8 -*-
# @Time    : 2023/9/11 18:31
# @Function:
import os
import re
import pickle
import pandas as pd
from lib.data_loader.process_data import pkl_to_df,pkl_loader
from torch_geometric.data import Data, InMemoryDataset
import torch
from glob2 import glob
from mne.io import RawArray, read_raw_edf
import dgl
import numpy as np
from collections import Counter

# from src.timeSeries.KVs_dyG import KVs_dyG


class chbmitDataset(InMemoryDataset):
    def __init__(self,cfg, root,transform=None, pre_transform=None):
        self.cfg = cfg
        self.length = cfg.data.ts_length
        self.l_min = cfg.data.l_min
        self.shapelet = True
        self.channels_name = ['FP1-F7', 'P7-O1', 'F3-C3', 'C3-P3', 'FP2-F8', 'P8-O2', 'F4-C4', 'C4-P4', 'T7-FT9', 'FT10-T8']
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
        # edf_file_list = glob(self.raw_dir + "/*.edf")
        # graph_num = len(edf_file_list)
        data_list = []
        label_dict = self.label_loader()
        value_counts =  Counter(label_dict.values())
        # print(value_counts)
        nodes = list(range(len(self.channels_name)))
        edge_index = torch.combinations(torch.tensor(nodes), 2).t()
        print(f"edge_num:{len(edge_index[0])}")
        self.graph_num = len(label_dict)
        if os.path.exists(os.path.join(self.processed_dir, 'all_fea_matrix.pkl')) and \
                os.path.exists(os.path.join(self.processed_dir, 'node_labels.pkl')):
            # all_fea_matrix = pkl_loader(os.path.join(self.processed_dir, 'all_fea_matrix.pkl'))
            channel_fea_dict = pkl_loader(os.path.join(self.processed_dir, 'channel_fea_dict.pkl'))
            node_labels = pkl_loader(os.path.join(self.processed_dir, 'node_labels.pkl'))

        else:

            edf_file_list = glob(self.raw_dir + "/*.edf")
            edf_file_list.sort()

            all_fea_matrix = []
            channel_fea_dict = {channel: [] for channel in self.channels_name}
            node_labels = []
            for path in edf_file_list:
                id = path.split('/')[-1].split('.')[0]
                recDf = read_raw_edf(path).to_data_frame()  # 读取单个edf文件
                if len(recDf) >= self.length:
                    y = label_dict[id]
                    recDf = recDf.head(self.length)
                    recDf = recDf[self.channels_name]
                    all_fea_matrix.append(recDf.values.tolist())
                    for i in self.channels_name:
                        channel_fea_dict[i].append(recDf[i].values.tolist())
                        node_labels.append(y)
            # store x and label
            pickle.dump(all_fea_matrix, open(os.path.join(self.processed_dir, 'all_fea_matrix.pkl'), 'wb'))
            pickle.dump(channel_fea_dict, open(os.path.join(self.processed_dir, 'channel_fea_dict.pkl'), 'wb'))
            pickle.dump(node_labels, open(os.path.join(self.processed_dir, 'node_labels.pkl'), 'wb'))

        print(f"graph_number:{len(node_labels)/10}")
        feas = KVs_dyG(self.cfg, channel_fea_dict, node_labels, self.processed_dir)

        for y in label_dict.values():
            y = torch.tensor(y)
            node_label = node_labels[:10]
            node_labels = node_labels[10:]
            x = feas[:10]
            feas =  feas[10:]
            u, v = edge_index[0], edge_index[1]
            g = dgl.graph((u, v))
            x = x[:g.num_nodes()]
            g.ndata['feat'] = x.float()

            data_list.append(Data(graph=g, x=x, y=y,  node_label=node_label, edge_index=edge_index, num_class=len(value_counts)))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def label_loader(self):
        '''
        :param path: file path
        :return: label of record (SEIZURES:1, Normal:0)
        (dataframe: {"ID":id,"Label":label})
        '''
        # if os.path.exists(raw_dir+'/'+"label.pkl"):
        #     label_df = pkl_to_df(raw_dir+'/'+"label.pkl")
        # else:
        allIDPath = 'RECORDS'
        seiIDPath = 'RECORDS-WITH-SEIZURES'
        allIDfile = pd.read_csv(os.path.join(self.raw_dir, allIDPath), header=None).values
        seiIDfile = pd.read_csv(os.path.join(self.raw_dir, seiIDPath), header=None).values
        # print(allID.shape)
        # label_df = pd.DataFrame()
        seiID = []
        names,ids,labels = [],[],[]
        label_dict = {}
        for row in seiIDfile:
            row = str(row)[2:-2]  # 去除前[' 后']
            sei = re.split('[/.]', row)  # 按'.'和'/'切割
            # patientID.append(sei[0])
            seiID.append(sei[1])
        for row in allIDfile:
            row = str(row)[2:-2]
            name = re.split('[/.]', row)
            id = name[1]
            label = 0
            if id in seiID: # 发病ts标签置1
                label = 1
            names += [name[0]]
            label_dict.update({id: label})
            # ids += [id]
            # labels +=  [label]
            # ser = pd.Series({"PatientID":name[0],"RecordID": id, "Label": label})


        # label_df = pd.DataFrame({"RecordID": ids, "Label": labels})
        # label_df.to_csv(process_dir + "label.csv", index=False, header=True)  # csv
        pickle.dump(label_dict, open(os.path.join(self.processed_dir, 'label.pkl'), 'wb'))  # pkl
        print('--------- label dataset: label.pkl saved! ---------')
        return label_dict



