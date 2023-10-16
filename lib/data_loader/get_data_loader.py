# -*- coding: utf-8 -*-
# @Time    : 2023/8/15 17:50
# @Function:
import torch
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from lib.data_loader import SynGraphDataset, Mutag, SPMotif,graph_sst2
from ogb.graphproppred import PygGraphPropPredDataset
from lib.data_loader.EEG_chbmit import chbmitDataset
from lib.data_loader.metr import metrDataset
def get_data_loaders(cfg,  splits, shapelet = False, mutag_x=False):
    multi_label = False
    l_min = 10
    dataset = None

    # if cfg.data.data_name in ['ba_2motifs', 'mutag', 'Graph-SST2', 'mnist',
    #                         'spmotif_0.5', 'spmotif_0.7', 'spmotif_0.9',
    #                         'ogbg_molhiv', 'ogbg_moltox21', 'ogbg_molbace',
    #                         'ogbg_molbbbp', 'ogbg_molclintox', 'ogbg_molsider']:
    if cfg.data.data_name in ['ba_2motifs', 'mutag', 'Graph-SST2', 'mnist',
                              'spmotif_0.5', 'spmotif_0.7', 'spmotif_0.9',
                              'ogbg_molhiv',  'ogbg_molbace',
                              'ogbg_molbbbp', 'ogbg_molclintox', 'ogbg_molsider']:
        data_dir =cfg.data.root+'dglData'
    elif 'chbmit' in cfg.data.data_name:
        data_dir =cfg.data.root +'chbmit'+ cfg.data.data_name
    else:
        data_dir =cfg.data.root + cfg.data.data_name

    if cfg.data.data_name == 'ba_2motifs':
        dataset = SynGraphDataset(data_dir, 'ba_2motifs',cfg, multi_label, l_min,shapelet=shapelet)
        split_idx = get_random_split_idx(dataset, splits)
        loaders, test_set = get_loaders_and_test_set(cfg.data['loader_size'], dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx["train"]]

    elif cfg.data.data_name == 'mutag':
        dataset = Mutag(cfg,  multi_label,l_min,root=data_dir+'/mutag',shapelet=shapelet)
        split_idx = get_random_split_idx(dataset, splits, mutag_x=mutag_x)
        loaders, test_set = get_loaders_and_test_set(cfg.data['loader_size'], dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx['train']]


    elif 'ogbg' in cfg.data.data_name:
        dataset = PygGraphPropPredDataset( name='-'.join(cfg.data.data_name.split('_')),root=data_dir)
        split_idx = dataset.get_idx_split()
        print('[INFO] Using default splits!')
        loaders, test_set = get_loaders_and_test_set(cfg.data['loader_size'], dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx['train']]
    #
    # elif cfg.data.data_name == 'Graph-SST2':
    #     dataset = graph_sst2.get_dataset(dataset_dir=data_dir, dataset_name='Graph-SST2', task=None)
    #     dataloader, (train_set, valid_set, test_set) = graph_sst2.get_dataloader(dataset, batch_size=cfg.data['loader_size'], degree_bias=True, seed=random_state)
    #     print('[INFO] Using default splits!')
    #     loaders = {'train': dataloader['train'], 'valid': dataloader['eval'], 'test': dataloader['test']}
    #     test_set = dataset  # used for visualization

    elif 'spmotif' in cfg.data.data_name:
        b = float(cfg.data.data_name.split('_')[-1])
        train_set = SPMotif(b,'train', cfg, multi_label,l_min,root=data_dir + '/'+ cfg.data.data_name,shapelet=shapelet )
        valid_set = SPMotif( b,'val',cfg, multi_label,l_min, root=data_dir+ '/' + cfg.data.data_name,shapelet=shapelet)
        test_set = SPMotif(b, 'test',cfg, multi_label,l_min,root=data_dir+ '/' + cfg.data.data_name, shapelet=shapelet)
        print('[INFO] Using default splits!')
        # for data in train_set:
        #     print(data)
        loaders, dataset = get_loaders_and_test_set(cfg.data.loader_size, dataset_splits={'train': train_set, 'valid': valid_set, 'test': test_set})
        # test = loaders['train']
        # for data in loaders['train']:
        #     print(data)

    elif 'chbmit' in cfg.data.data_name:
        # labels = label_loader(data_dir)
        dataset = chbmitDataset(cfg, root=data_dir)
        split_idx = get_random_split_idx(dataset, splits, mutag_x=mutag_x)
        loaders, test_set = get_loaders_and_test_set(cfg.data['loader_size'], dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx['train']]

    elif cfg.data.data_name == 'metr':
        dataset = metrDataset(cfg, root=data_dir)
        split_idx = get_random_split_idx(dataset, splits, mutag_x=mutag_x)
        loaders, test_set = get_loaders_and_test_set(cfg.data['loader_size'], dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx['train']]
        print('')



    # elif cfg.data.data_name == 'mnist':
    #     n_train_data, n_val_data = 20000, 5000
    #     train_val = MNIST75sp(data_dir / 'mnist', mode='train')
    #     perm_idx = torch.randperm(len(train_val), generator=torch.Generator().manual_seed(random_state))
    #     train_val = train_val[perm_idx]
    #
    #     train_set, valid_set = train_val[:n_train_data], train_val[-n_val_data:]
    #     test_set = MNIST75sp(data_dir / 'mnist', mode='test')
    #     loaders, test_set = get_loaders_and_test_set(cfg.data['loader_size'], dataset_splits={'train': train_set, 'valid': valid_set, 'test': test_set})
    #     print('[INFO] Using default splits!')

    # x_dim = test_set[0].x.shape[1]
    edge_attr_dim = 0 if test_set[0].edge_attr is None else test_set[0].edge_attr.shape[1]
    if isinstance(test_set, list):
        num_class = Batch.from_data_list(test_set).y.unique().shape[0]
    elif test_set.data.y.shape[-1] == 1 or len(test_set.data.y.shape) == 1:
        num_class = test_set.data.y.unique().shape[0]
    else:
        num_class = test_set.data.y.shape[-1]
    # int_tensor_list = test_set.data.edge_label.long()
    # print(torch.bincount( int_tensor_list))

    node_feat_dim = dataset.data.x.shape[1]

    print('[INFO] Calculating degree...')
    # Compute in-degree histogram over training data.
    # deg = torch.zeros(10, dtype=torch.long)
    # for data in train_set:
    #     d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    #     deg += torch.bincount(d, minlength=deg.numel())
    batched_train_set = Batch.from_data_list(train_set)
    d = degree(batched_train_set.edge_index[1], num_nodes=batched_train_set.num_nodes, dtype=torch.long)
    deg = torch.bincount(d, minlength=10)

    aux_info = {'deg': deg, 'multi_label': multi_label,'node_feat_dim':node_feat_dim,
                'edge_attr_dim':edge_attr_dim,'num_class':num_class,'l_min':l_min }
    return loaders, dataset,test_set,aux_info

def get_random_split_idx(dataset, splits, random_state=None, mutag_x=False):
    if random_state is not None:
        np.random.seed(random_state)

    print('[INFO] Randomly split dataset!')
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    if not mutag_x:
        n_train, n_valid = int(splits['train'] * len(idx)), int(splits['valid'] * len(idx))
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train+n_valid]
        test_idx = idx[n_train+n_valid:]
    else:
        print('[INFO] mutag_x is True!')
        n_train = int(splits['train'] * len(idx))
        train_idx, valid_idx = idx[:n_train], idx[n_train:]
        test_idx = [i for i in range(len(dataset)) if (dataset[i].y.squeeze() == 0 and dataset[i].edge_label.sum() > 0)]
    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}


def get_loaders_and_test_set(batch_size, dataset=None, split_idx=None, dataset_splits=None):
    # for data in dataset_splits['train']:
    #     print(data)
    if split_idx is not None:
        assert dataset is not None
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)
        test_set = dataset.copy(split_idx["test"])  # For visualization
    else:
        assert dataset_splits is not None
        train_loader = DataLoader(dataset_splits['train'], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset_splits['valid'], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset_splits['test'], batch_size=batch_size, shuffle=False)
        test_set = dataset_splits['test']  # For visualization

    # for data in test_loader:
    #     print(data)

    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}, test_set
