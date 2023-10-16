# -*- coding: utf-8 -*-
# @Time    : 2023/8/16 11:25
# @Function:
import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric.utils as utils
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from tslearn.preprocessing import TimeSeriesScalerMinMax
from torch_geometric.utils import add_self_loops
def process_edge(data, use_edge_attr):
    if not use_edge_attr:
        data.edge_attr = None
    if data.get('edge_label', None) is None:
        data.edge_label = torch.zeros(data.edge_index.shape[1])
    return data

def pkl_to_df(path):
    with open(path, "rb") as f:
        res = pickle.load(f)
    df = pd.DataFrame(res)
    return df

def pkl_loader(path):
    with open(path, "rb") as f:
        res = pickle.load(f)
    return res


def data_normal_process(f_channel,args):
    if args.sampling.samplingFlag == True:
        f_channel = [sublist[::args.sampling.step] for sublist in f_channel]

        # f_channel =  f_channel[::args.sampling.step]
    if args.sampling.normalizeFlag == True:
        data_array = np.array(f_channel)
        scaler = MinMaxScaler()
        f_channel = scaler.fit_transform( data_array)
        # f_channel = normalized_data.flatten()

        # f_channel =min_max_scaling(f_channel)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # f_channel = min_max_scaler.fit_transform(f_channel)
    return f_channel
#
# def min_max_scaling(my_list):
#     min_val = min(my_list)
#     max_val = max(my_list)
#     normalized_list = [(x - min_val) / (max_val - min_val) for x in my_list]
#     return normalized_list

def delete_edge(data,delete_r,pred_edge_logit):
    # 删除r%重要的边,保留1-r%不重要的边
    num_edges_to_keep = int(data.edge_attr.size(0) * (1-delete_r))

    # 选择不重要边的索引
    top_values, top_indices = torch.topk(-pred_edge_logit.view(-1), num_edges_to_keep)

    # 删除边
    edge_index = data.edge_index[:,top_indices ]
    edge_attr =  data.edge_attr[top_indices ]
    new_data = Data(x= data.x, y=data.y, edge_index = edge_index,edge_attr = edge_attr, batch = data.batch)

    # new_data= add_self_loops(new_data )
    return  new_data

def retain_edge(data,retain_r,pred_edge_logit):
    #保留重要的边
    num_edges_to_keep = int(data.edge_attr.size(0) * (retain_r))
    # 根据权重值排序边
    top_values, top_indices = torch.topk(pred_edge_logit.view(-1), num_edges_to_keep)
    edge_index = data.edge_index[:, top_indices]
    edge_attr = data.edge_attr[top_indices]
    new_data = Data(x=data.x, y=data.y, edge_index=edge_index, edge_attr=edge_attr, batch = data.batch)
    # new_data = add_self_loops(new_data)
    return new_data


def InsGNN_subgraph(data,edge_score):
    bottom_indices = (edge_score.squeeze() > 0.5).nonzero().squeeze()
    edge_index = data.edge_index[:, bottom_indices]
    key_edgeattr_delete = data.edge_attr[bottom_indices]
    flips_data = Data(x=data.x, y=data.y, edge_index=edge_index, edge_attr= key_edgeattr_delete, batch=data.batch)

    top_indices = (edge_score.squeeze()> 0.5).nonzero().squeeze()
    edge_index = data.edge_index[:, top_indices]
    key_edgeattr_maintain =  data.edge_attr[top_indices]
    retain_data = Data(x=data.x, y=data.y, edge_index=edge_index, edge_attr= key_edgeattr_maintain , batch=data.batch)

    return flips_data,retain_data
