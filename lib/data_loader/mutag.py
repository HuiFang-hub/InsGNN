# https://github.com/flyingdoog/PGExplainer/blob/master/MUTAG.ipynb

import yaml
import torch
import numpy as np
import pickle as pkl
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
from src.timeSeries.KVs_staticG import KVs_staticG
import dgl

class Mutag(InMemoryDataset):
    def __init__(self, cfg,  multi_label,l_min,root,shapelet=False):
        self.shapelet = shapelet
        self.ag = cfg
        # self.use_edge_attr = use_edge_attr
        self.multi_label = multi_label
        self.l_min = l_min
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Mutagenicity_A.txt', 'Mutagenicity_edge_gt.txt', 'Mutagenicity_edge_labels.txt',
                'Mutagenicity_graph_indicator.txt', 'Mutagenicity_graph_labels.txt', 'Mutagenicity_label_readme.txt',
                'Mutagenicity_node_labels.txt', 'Mutagenicity.pkl']

    @property
    def processed_file_names(self):
        if self.shapelet:
            return ['shapelet_data.pt']
        else:
            return ['data.pt']

    def download(self):
        raise NotImplementedError

    def process(self):
        with open(self.raw_dir + '/Mutagenicity.pkl', 'rb') as fin:
            _, original_features, original_labels = pkl.load(fin)

        edge_lists, graph_labels, edge_label_lists, node_type_lists = self.get_graph_data()
        num_class = len(np.bincount(graph_labels))
        data_list = []
        pbar = tqdm(range(original_labels.shape[0]))
        for i in pbar:
            num_nodes = len(node_type_lists[i])
            edge_index = torch.tensor(edge_lists[i], dtype=torch.long).T
            y = torch.tensor(graph_labels[i]).float().reshape(-1, 1)
            x = torch.tensor(original_features[i][:num_nodes]).float()
            assert original_features[i][num_nodes:].sum() == 0
            edge_label = torch.tensor(edge_label_lists[i]).float()
            if y.item() != 0:
                edge_label = torch.zeros_like(edge_label).float()

            node_label = torch.zeros(x.shape[0])
            signal_nodes = list(set(edge_index[:, edge_label.bool()].reshape(-1).tolist()))
            if y.item() == 0:
                node_label[signal_nodes] = 1

            if len(signal_nodes) != 0:
                node_type = torch.tensor(node_type_lists[i])
                node_type = set(node_type[signal_nodes].tolist())
                assert node_type in ({4, 1}, {4, 3}, {4, 1, 3})  # NO or NH

            if y.item() == 0 and len(signal_nodes) == 0:
                continue

            if self.shapelet :
                # data = process_data(data, use_edge_attr).to(device)
                # cfg.shaplet_outsize = data.x.shape[1]
                # print(f"node_label bincount:{torch.bincount(node_label.squeeze().int())}")

                # shapletE = Shapelet_static(self.ag, num_class, self.multi_label,self.l_min)
                # feature  = shapletE(x, node_label,)
                feature = KVs_staticG(self.ag, x, node_label, num_class)

                # feature = torch.from_numpy(np.array([[0.1] * x.shape[0]]).T)
                x = feature.cpu()
                # x = torch.concat([x, feature], dim=1)
                # num_nodes, node_feat_dim = x.shape[0], x.shape[1]
            # print(np.bincount(graph_labels))
            u, v = edge_index[0], edge_index[1]
            g = dgl.graph((u, v))
            x=x[:g.num_nodes()]
            g.ndata['feat'] =x.float()
            # graph = g.to(self.cfg.device)

            data_list.append(Data(graph=g,x=x, y=y, edge_index=edge_index, node_label=node_label,
                                  edge_label=edge_label, node_type=torch.tensor(node_type_lists[i]),num_class=num_class))


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_graph_data(self):
        pri = self.raw_dir + '/Mutagenicity_'

        file_edges = pri + 'A.txt'
        # file_edge_labels = pri + 'edge_labels.txt'
        file_edge_labels = pri + 'edge_gt.txt'
        file_graph_indicator = pri + 'graph_indicator.txt'
        file_graph_labels = pri + 'graph_labels.txt'
        file_node_labels = pri + 'node_labels.txt'

        edges = np.loadtxt(file_edges, delimiter=',').astype(np.int32)
        try:
            edge_labels = np.loadtxt(file_edge_labels, delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use edge label 0')
            edge_labels = np.zeros(edges.shape[0]).astype(np.int32)

        graph_indicator = np.loadtxt(file_graph_indicator, delimiter=',').astype(np.int32)
        graph_labels = np.loadtxt(file_graph_labels, delimiter=',').astype(np.int32)

        try:
            node_labels = np.loadtxt(file_node_labels, delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use node label 0')
            node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32)

        graph_id = 1
        starts = [1]
        node2graph = {}
        for i in range(len(graph_indicator)):
            if graph_indicator[i] != graph_id:
                graph_id = graph_indicator[i]
                starts.append(i+1)
            node2graph[i+1] = len(starts)-1
        # print(starts)
        # print(node2graph)
        graphid = 0
        edge_lists = []
        edge_label_lists = []
        edge_list = []
        edge_label_list = []
        for (s, t), l in list(zip(edges, edge_labels)):
            sgid = node2graph[s]
            tgid = node2graph[t]
            if sgid != tgid:
                print('edges connecting different graphs, error here, please check.')
                print(s, t, 'graph id', sgid, tgid)
                exit(1)
            gid = sgid
            if gid != graphid:
                edge_lists.append(edge_list)
                edge_label_lists.append(edge_label_list)
                edge_list = []
                edge_label_list = []
                graphid = gid
            start = starts[gid]
            edge_list.append((s-start, t-start))
            edge_label_list.append(l)

        edge_lists.append(edge_list)
        edge_label_lists.append(edge_label_list)

        # node labels
        node_label_lists = []
        graphid = 0
        node_label_list = []
        for i in range(len(node_labels)):
            nid = i+1
            gid = node2graph[nid]
            # start = starts[gid]
            if gid != graphid:
                node_label_lists.append(node_label_list)
                graphid = gid
                node_label_list = []
            node_label_list.append(node_labels[i])
        node_label_lists.append(node_label_list)

        return edge_lists, graph_labels, edge_label_lists, node_label_lists
