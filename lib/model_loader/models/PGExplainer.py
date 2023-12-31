from typing import Optional
from math import sqrt
from torch_geometric.utils.num_nodes import maybe_num_nodes
import os
import time
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_networkx
# from Configures import model_args, data_args, explainer_args
from dgl import DGLGraph
EPS = 1e-6


class PGExplainer(nn.Module):
    def __init__(self, model, cfg,model_path, top_k: int = 6, num_hops: Optional[int] = None):
        # lr=0.005, 0.003
        super(PGExplainer, self).__init__()
        explainer_args = cfg.pgexplainer.explainer_args
        model_args = cfg.pgexplainer.model_args
        self.cfg = cfg
        self.model = model
        self.lr = cfg.train.graph_lr
        self.epochs = cfg.train.epochs
        self.top_k = top_k
        self.__num_hops__ = num_hops
        self.device = model.device

        self.coff_size =  explainer_args.coff_size
        self.coff_ent = explainer_args.coff_ent
        self.init_bias = 0.0
        self.t0 = explainer_args.t0
        self.t1 = explainer_args.t1

        self.elayers = nn.ModuleList()
        if model_args.model_name == 'gat':
            input_feature = model_args.gat_heads * model_args.gat_hidden * 2
        elif model_args.concate:
            input_feature = int(torch.Tensor(model_args.latent_dim).sum()) * 2
        elif model_args.model_name == 'devign':
            input_feature = 100 * 2
        else:
            input_feature = model_args.latent_dim[-1] * 2
        self.elayers.append(nn.Sequential(nn.Linear(input_feature, 64), nn.ReLU()))
        self.elayers.append(nn.Linear(64, 1))
        self.elayers.to(self.device)
        self.ckpt_path = os.path.join(model_path,f'PGE_generator_{model_args.model_name}.ckpt')

    def __set_masks__(self, x, edge_index, edge_mask=None, init="normal"):
        """ Set the weights for message passing """
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1
        init_bias = self.init_bias
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * std)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))

        if edge_mask is None:
            self.edge_mask = torch.randn(E) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask.to(self.device)
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        """ clear the edge weights to None """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None

    @property
    def num_hops(self):
        """ return the number of layers of GNN model """
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __loss__(self, prob, ori_pred):
        """
        the pred loss encourages the masked graph with higher probability,
        the size loss encourage small size edge mask,
        the entropy loss encourage the mask to be continuous.
        """
        logit = prob[ori_pred]
        logit = logit + EPS
        pred_loss = -torch.log(logit)
        # size
        edge_mask = torch.sigmoid(self.mask_sigmoid)
        size_loss = self.coff_size * torch.sum(edge_mask)

        # entropy
        edge_mask = edge_mask * 0.99 + 0.005
        mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
        mask_ent_loss = self.coff_ent * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss
        return loss

    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """ Sample from the instantiation of concrete distribution when training
        \epsilon \sim  U(0,1), \hat{e}_{ij} = \sigma (\frac{\log \epsilon-\log (1-\epsilon)+\omega_{i j}}{\tau})
        """
        if training:
            random_noise = torch.rand(log_alpha.shape)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def forward(self, inputs, training=None):
        x, embed, edge_index, tmp, edge_attr= inputs
        nodesize = embed.shape[0]
        feature_dim = embed.shape[1]
        f1 = embed.unsqueeze(1).repeat(1, nodesize, 1).reshape(-1, feature_dim)
        f2 = embed.unsqueeze(0).repeat(nodesize, 1, 1).reshape(-1, feature_dim)

        # using the node embedding to calculate the edge weight
        f12self = torch.cat([f1, f2], dim=-1)
        h = f12self.to(self.device)
        for elayer in self.elayers:
            h = elayer(h)
        values = h.reshape(-1)
        values = self.concrete_sample(values, beta=tmp, training=training)
        self.mask_sigmoid = values.reshape(nodesize, nodesize)

        # set the symmetric edge weights
        sym_mask = (self.mask_sigmoid + self.mask_sigmoid.transpose(0, 1)) / 2
        edge_mask = sym_mask[edge_index[0], edge_index[1]]

        self.__clear_masks__()
        self.__set_masks__(x, edge_index, edge_mask)

        # the model prediction with edge mask
        #data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
        #data.to(self.device)
        graph = type_conversion(x, edge_index, edge_attr)
        outputs = self.model(graph, cuda=True)
        return outputs[1].squeeze(), edge_mask

    def get_model_output(self, x, edge_index, edge_attr, edge_mask=None, **kwargs):
        """ return the model outputs with or without (w/wo) edge mask  """
        self.model.eval()
        self.__clear_masks__()
        if edge_mask is not None:
            self.__set_masks__(x, edge_index, edge_mask.to(self.device))

        with torch.no_grad():
            #data = Batch.from_data_list([Data(x=x, edge_index=edge_index, edge_attr=edge_attr)])
            #data.to(self.device)
            graph = type_conversion(x, edge_index, edge_attr)
            outputs = self.model(graph, cuda=True)

        self.__clear_masks__()
        return outputs

    def train_GC_explanation_network(self, dataset):
        """ train the explantion network for graph classification task """
        optimizer = Adam(self.elayers.parameters(), lr=self.lr)
        if self.cfg.data.dataset_name.lower() == 'grt_sst2_BERT_Identity'.lower():
            split_indices = dataset.supplement['split_indices']
            dataset_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        else:
            dataset_indices = list(range(len(dataset)))

        # collect the embedding of nodes
        emb_dict = {}
        ori_pred_dict = {}
        with torch.no_grad():
            self.model.eval()
            for gid in tqdm(dataset_indices):
                data = dataset[gid]
                _, prob, emb = self.get_model_output(data.x, data.edge_index, data.edge_attr)
                emb_dict[gid] = emb.data.cpu()
                ori_pred_dict[gid] = prob.argmax(-1).data.cpu()

        # train the mask generator
        duration = 0.0
        for epoch in range(self.epochs):
            loss = 0.0
            tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
            self.elayers.train()
            optimizer.zero_grad()
            tic = time.perf_counter()
            for gid in tqdm(dataset_indices):
                data = dataset[gid]
                prob, _ = self.forward((data.x, emb_dict[gid], data.edge_index, tmp, data.edge_attr), training=True)
                loss_tmp = self.__loss__(prob, ori_pred_dict[gid])
                loss_tmp.backward()
                loss += loss_tmp.item()

            optimizer.step()
            duration += time.perf_counter() - tic
            print(f'Epoch: {epoch} | Loss: {loss}')
            torch.save(self.elayers.cpu().state_dict(), self.ckpt_path)
            self.elayers.to(self.device)
        print(f"training time is {duration:.5}s")

    def get_explanation_network(self, dataset, is_graph_classification=True):
        if os.path.isfile(self.ckpt_path):
            print("fetch network parameters from the saved files")
            state_dict = torch.load(self.ckpt_path)
            self.elayers.load_state_dict(state_dict)
            self.to(self.device)
        elif is_graph_classification:
            self.train_GC_explanation_network(dataset)
        else:
            self.train_NC_explanation_network(dataset)

    def eval_probs(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                    edge_mask: torch.Tensor=None, **kwargs) -> None:
        outputs = self.get_model_output(x, edge_index, edge_attr, edge_mask=edge_mask)
        return outputs[1].squeeze()

    def explain_edge_mask(self, x, edge_index, edge_attr, **kwargs):
        with torch.no_grad():
            _, prob, emb = self.get_model_output(x, edge_index, edge_attr)
            _, edge_mask = self.forward((x, emb, edge_index, 1.0, edge_attr), training=False)
        return edge_mask

    def get_subgraph(self, node_idx, x, edge_index, y, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        graph = to_networkx(data=Data(x=x, edge_index=edge_index), to_undirected=True)

        subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        mapping = {int(v): k for k, v in enumerate(subset)}
        subgraph = graph.subgraph(subset.tolist())
        nx.relabel_nodes(subgraph, mapping)

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item
        y = y[subset]
        return x, edge_index, y, subset, kwargs

    def train_NC_explanation_network(self, dataset):
        data = dataset[0]
        dataset_indices = torch.where(data.train_mask != 0)[0].tolist()
        optimizer = Adam(self.elayers.parameters(), lr=self.lr)

        # collect the embedding of nodes
        x_dict = {}
        edge_index_dict = {}
        node_idx_dict = {}
        emb_dict = {}
        pred_dict = {}
        with torch.no_grad():
            self.model.eval()
            for gid in dataset_indices:
                x, edge_index, y, subset, _ = \
                    self.get_subgraph(node_idx=gid, x=data.x, edge_index=data.edge_index, y=data.y)
                _, prob, emb = self.get_model_output(x, edge_index)

                x_dict[gid] = x
                edge_index_dict[gid] = edge_index
                node_idx_dict[gid] = int(torch.where(subset == gid)[0])
                pred_dict[gid] = prob[node_idx_dict[gid]].argmax(-1).cpu()
                emb_dict[gid] = emb.data.cpu()

        # train the explanation network
        for epoch in range(self.epochs):
            loss = 0.0
            optimizer.zero_grad()
            tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
            self.elayers.train()
            for gid in tqdm(dataset_indices):
                pred, edge_mask = self.forward((x_dict[gid], emb_dict[gid], edge_index_dict[gid], tmp), training=True)
                loss_tmp = self.__loss__(pred[node_idx_dict[gid]], pred_dict[gid])
                loss_tmp.backward()
                loss += loss_tmp.item()

            optimizer.step()
            print(f'Epoch: {epoch} | Loss: {loss}')
            torch.save(self.elayers.cpu().state_dict(), self.ckpt_path)
            self.elayers.to(self.device)

    def eval_node_probs(self, node_idx: int, x: torch.Tensor,
                        edge_index: torch.Tensor, edge_mask: torch.Tensor, **kwargs):
        probs = self.eval_probs(x=x, edge_index=edge_index, edge_mask=edge_mask, **kwargs)
        return probs[node_idx].squeeze()

    def get_node_prediction(self, node_idx: int, x: torch.Tensor, edge_index: torch.Tensor, **kwargs):
        outputs = self.get_model_output(x, edge_index, edge_mask=None, **kwargs)
        return outputs[1][node_idx].argmax(dim=-1)

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def type_conversion(x, edge_index, edge_attr):
    graph = DGLGraph()
    x=x.cpu()
    graph.add_nodes(len(x), data={'features': torch.FloatTensor(x)})
    edge_index_list = edge_index.cpu().t().numpy().tolist()
    edge_attr_list = edge_attr.cpu().numpy().tolist()
    for i in range(len(edge_index_list)):
        graph.add_edges(edge_index_list[i][0], edge_index_list[i][1], data={'etype': torch.LongTensor([edge_attr_list[i][0]])})
    return graph

def k_hop_subgraph_with_default_whole_graph(node_idx, num_hops,
    edge_index, relabel_nodes=False, num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`. If the node_idx is -1, then return the original graph.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`. when num_hops == -1,
            the whole graph will be returned.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index  # edge_index 0 to 1, col: source, row: target

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    inv = None

    if int(node_idx) == -1:
        subsets = torch.tensor([0])
        cur_subsets = subsets
        while 1:
            node_mask.fill_(False)
            node_mask[subsets] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets = torch.cat([subsets, col[edge_mask]]).unique()
            if not cur_subsets.equal(subsets):
                cur_subsets = subsets
            else:
                subset = subsets
                break
    else:
        if isinstance(node_idx, (int, list, tuple)):
            node_idx = torch.tensor([node_idx], device=row.device, dtype=torch.int64).flatten()
        elif isinstance(node_idx, torch.Tensor) and len(node_idx.shape) == 0:
            node_idx = torch.tensor([node_idx])
        else:
            node_idx = node_idx.to(row.device)

        subsets = [node_idx]
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask