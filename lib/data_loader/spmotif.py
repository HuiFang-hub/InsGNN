# From Discovering Invariant Rationales for Graph Neural Networks
import pickle
import os.path as osp
import pickle as pkl
from tqdm import tqdm
import yaml
import torch
import torch.nn.functional as F
import random
import numpy as np
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data
from src.timeSeries import KVs_staticG
import dgl
import os
try:
    from .spmotif_utils import gen_dataset
except ImportError:
    from spmotif_utils import gen_dataset


class SPMotif(InMemoryDataset):
    splits = ['train', 'val', 'test']

    def __init__(self, b, mode,  cfg, multi_label,l_min,root, shapelet=False,transform=None, pre_transform=None, pre_filter=None):

        assert mode in self.splits
        self.b = b
        self.mode = mode
        self.shapelet = shapelet
        self.ag = cfg
        # self.use_edge_attr = use_edge_attr
        self.multi_label = multi_label
        self.l_min = l_min
        super(SPMotif, self).__init__(root, transform, pre_transform, pre_filter)
        if self.shapelet:
            idx = self.processed_file_names.index('shapelet_SPMotif_{}.pt'.format(mode))
        else:
            idx = self.processed_file_names.index('SPMotif_{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return ['train.npy', 'val.npy', 'test.npy']

    @property
    def processed_file_names(self):
        if self.shapelet:
            return ['shapelet_SPMotif_train.pt', 'shapelet_SPMotif_val.pt', 'shapelet_SPMotif_test.pt']
        else:

            return ['SPMotif_train.pt', 'SPMotif_val.pt', 'SPMotif_test.pt']

    def download(self):
        print('[INFO] Generating SPMotif dataset...')
        gen_dataset(self.b, Path(self.raw_dir))

    def process(self):

        idx = self.raw_file_names.index('{}.npy'.format(self.mode))
        edge_index_list, label_list, ground_truth_list, role_id_list, pos = np.load(osp.join(self.raw_dir, self.raw_file_names[idx]), allow_pickle=True)
        data_list = []
        f = []
        # num_class = len(np.bincount(label_list))
        pbar = tqdm(zip(edge_index_list, label_list, ground_truth_list, role_id_list, pos))
        for idx, (edge_index, y, ground_truth, z, p) in enumerate(pbar):
            edge_index = torch.from_numpy(edge_index).long()
            node_idx = torch.unique(edge_index)
            assert node_idx.max() == node_idx.size(0) - 1
            # x = torch.zeros(node_idx.size(0), 4)
            # index = [i for i in range(node_idx.size(0))]
            # x[index, z] = 1
            x = torch.rand((node_idx.size(0), 4))
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.tensor(y, dtype=torch.long).reshape(-1)

            node_label = torch.tensor(z, dtype=torch.float)
            node_label[node_label != 0] = 1
            edge_label = torch.tensor(ground_truth.tolist(), dtype=torch.float)
            num_class = len(np.bincount(label_list.astype(int)))
            if self.shapelet :
                # data = process_data(data, use_edge_attr).to(device)
                # cfg.shaplet_outsize = data.x.shape[1]
                # print(f"node_label bincount:{torch.bincount(node_label.squeeze().int())}")
                feature = KVs_staticG(self.ag, x, node_label, num_class, self.multi_label,self.l_min).cpu()
                # feature = torch.from_numpy(np.array([[0.1] * x.shape[0]]).T)
                x = feature
                f.append(x)
                # x = torch.concat([x, feature], dim=1)
                # num_nodes, node_feat_dim = x.shape[0], x.shape[1]
            # print(np.bincount(graph_labels))

            u, v = edge_index[0], edge_index[1]
            g = dgl.graph((u, v))
            x = x[:g.num_nodes()]
            g.ndata['feat'] = x.float()
            data = Data(graph=g,x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, node_label=node_label, edge_label=edge_label,num_class=num_class)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        #
        if self.shapelet:
            f = torch.cat(f)
            idx = self.processed_file_names.index('shapelet_SPMotif_{}.pt'.format(self.mode))
            pickle.dump(f,  open(os.path.splitext(str(self.processed_paths[idx]))[0] +'_shaplets.pkl', 'wb'))
            torch.save(self.collate(data_list), self.processed_paths[idx])
        else:
            idx = self.processed_file_names.index('SPMotif_{}.pt'.format(self.mode))
            torch.save(self.collate(data_list), self.processed_paths[idx])
        #
