# -*- coding: utf-8 -*-
# @Time    : 2023/9/4 19:38
# @Function:
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_add_pool
from lib.data_loader.process_data import process_edge
from tqdm import tqdm
from copy import deepcopy
import dgl
import numpy as np
import logging
from log.save_results import save_res,save_model
from lib.result_comp import get_preds,result_compute
from torch_geometric.nn import BatchNorm, global_mean_pool
from lib.data_loader.process_data import process_edge, delete_edge, retain_edge
from sklearn.metrics import accuracy_score
from torch.nn import Sequential, ReLU, Linear
class GATClassifier(nn.Module):
    def __init__(self, num_features, num_class,multi_label,model_config,in_head = 8, out_head = 1,dropout= 0.6):
        super(GATClassifier, self).__init__()
        hidden_size = model_config['hidden_size']
        self.conv1 = GATConv(num_features, 8, heads=in_head, dropout=dropout,add_self_loops = False)
        self.conv2 = GATConv(8 * 8, hidden_size, heads=out_head, dropout=0.6,add_self_loops = False)

        self.fc_out = Sequential(Linear(hidden_size, hidden_size // 2), ReLU(),
                                 Linear(hidden_size // 2, hidden_size // 4), ReLU(),
                                 Linear(hidden_size // 4, 1 if num_class == 2 and not multi_label else num_class))

        self.pool = global_mean_pool
    def forward(self, data):
        edge_index, batch = data.edge_index, data.batch

        x = F.relu(self.conv1(data.x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x, (edge_index, alpha) = self.conv2(x, edge_index, return_attention_weights=True)
        # x = global_add_pool(x, batch)  # 对节点特征进行汇总
        x = self.pool(x, batch)
        return self.fc_out(x),alpha


# class GATv3(nn.Module):
#     #linear layer and GAT layer have the same shape of in-out features
#     def __init__(self,
#                  num_layers,
#                  input_dim,
#                  hidden_dim,
#                  num_classes,
#                  heads,
#                  activation,
#                  feat_drop,
#                  attn_drop,
#                  negative_slope,
#                  residual,
#                  final_dropout,
#                  graph_pooling_type,
#                  linear_num_layers):
#         super(GATv3, self).__init__()
#         self.num_layers = num_layers
#         self.gatv3_layers = nn.ModuleList()
#         self.batch_norms = nn.ModuleList()
#         self.linears_prediction = torch.nn.ModuleList()
#         self.activation = activation
#         self.gatv3_layers.append(GATv2Conv(
#             input_dim, hidden_dim, heads[0],
#             feat_drop, attn_drop, negative_slope, False, self.activation,
#             bias=False, share_weights=True,allow_zero_in_degree=True))
#         self.batch_norms.append(nn.BatchNorm1d(hidden_dim*heads[0]))
#         # hidden layers
#         for l in range(1, self.num_layers):
#             # due to multi-head, the in_dim = num_hidden * num_heads
#             self.gatv3_layers.append(GATv2Conv(
#                 hidden_dim * heads[l-1 ], hidden_dim, heads[l],
#                 feat_drop, attn_drop, negative_slope, residual,
#                 self.activation, bias=False, share_weights=True,allow_zero_in_degree=True))
#             self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads[l]))
#         self.output_dim = num_classes
#         self.linears_prediction = torch.nn.ModuleList()
#
#         for layer in range(num_layers):
#             if layer==0:
#                 self.linears_prediction.append( nn.Linear(input_dim,self.output_dim))
#             else:
#                 self.linears_prediction.append(nn.Linear( hidden_dim * heads[layer-1],self.output_dim))
#         self.linears_prediction.append(nn.Linear(hidden_dim * heads[num_layers-1], self.output_dim))
#         self.drop = nn.Dropout(final_dropout)
#         self.pool = global_add_pool
#
#         self.fc_out = nn.Sequential(nn.Linear(hidden_dim, 2))
#
#     def forward(self, g,batch):
#         h = g.ndata['feat']
#         # eh = g.edata['feat']
#         hidden_rep = [h]
#         score = None
#         for l in range(self.num_layers):
#             # if l == self.num_layers-1: #最后一层没有batch normal
#             h,score = self.gatv3_layers[l](g,h,get_attention=True)
#             # h = h
#             h = h.flatten(1)
#             h = self.batch_norms[l](h)
#             h = F.relu(h)
#             hidden_rep.append(h)
#         # output projection
#         score_over_layer = 0
#         for i, h in enumerate(hidden_rep):
#             pooled_h = self.pool(h, batch)
#             score_over_layer += self.linears_prediction[i](pooled_h)
#         # logits = self.gatv2_layers[-1](g, h).mean(1)
#         return score_over_layer,score

class gat_model(nn.Module):
    def __init__(self,cfg, aux_info):
        super().__init__()

        self.node_feat_dim, self.num_class = aux_info['node_feat_dim'], aux_info['num_class']
        self.cfg = cfg
        self.heads = ([cfg.gat.num_heads] * (cfg.model.num_layers - 1)) + [cfg.gat.num_out_heads]

        multi_label =  aux_info['multi_label']

        # self.model = GATv3(num_layers=cfg.model.num_layers,
        #                    input_dim=self.node_feat_dim,  # args.shaplet_outsize,#node_feat_dim,
        #                    hidden_dim=cfg.model.hidden_size,
        #                    num_classes=self.num_class,
        #                    heads=self.heads,
        #                    activation=F.elu,
        #                    feat_drop=cfg.model.dropout_p,
        #                    attn_drop=cfg.gat.attn_drop,
        #                    negative_slope=cfg.gat.negative_slope,
        #                    residual=cfg.gat.residual,
        #                    final_dropout=cfg.gat.final_dropout,
        #                    graph_pooling_type=cfg.gat.graph_pooling_type,
        #                    linear_num_layers=cfg.model.num_mlp_layers
        #                    ).to(cfg.device)
        self.model = GATClassifier(self.node_feat_dim, self.num_class,multi_label,cfg.model,
                                   in_head = 8, out_head = 1,dropout = cfg.model.dropout_p).to(cfg.device)

        self.model_dir = cfg.model.model_save_path
        self.use_edge_attr = cfg.shared_config.use_edge_attr
        self.init_metric_dict = {'epoch': 0, 'acc': 0, 'roc_auc': 0}
        self.seed = cfg.seed
        self.multi_class = aux_info['multi_label']

    def eval_one_batch(self, data, epoch=None):
        # batch = data.batch
        # # u, v = data.edge_index[0], data.edge_index[1]
        # # g = dgl.graph((u, v))
        # # g.ndata['feat'] = data.x.float()
        # graph = data.graph.to(self.cfg.device)
        # pred_g_logit, pred_edge_logit = self.model(graph, batch)
        with torch.no_grad():
            pred_g_logit, pred_edge_logit = self.model(data)
        # pred_edge_logit = self.model.conv1.alpha
        return pred_edge_logit, pred_g_logit, 0

    def train_one_batch(self, data,  epoch=None):
        # self.model.train()
        # label = data.y.squeeze()
        # batch = data.batch
        # # u, v = data.edge_index[0], data.edge_index[1]
        # # g = dgl.graph((u, v))
        # # g.ndata['feat'] = data.x.float()
        # # graph = data.graph
        # pred_g_logit = self.model(data)
        # train_loss = self.criterion(pred_g_logit, label.long())
        # self.optimizer.zero_grad()
        # train_loss.backward()
        # self.optimizer.step()
        # self.model.eval()
        # pred_edge_logit = self.model.conv1.att(data.x, data.edge_index)
        # self.model.train()

        with torch.no_grad():
            pred_g_logit,pred_edge_logit = self.model(data)
        # pred_edge_logit = self.model.conv1.att

        return pred_edge_logit, pred_g_logit, 0

    def run_one_epoch(self, data_loader, cur_mode):
        run_one_batch = self.train_one_batch if cur_mode == 'train' else self.eval_one_batch

        # loss = 0
        edge_true, edge_score = [], []
        g_true, g_score = [], []
        pbar = tqdm(data_loader)
        for idx, data in enumerate(pbar):
            data = process_edge(data, self.use_edge_attr)
            data = data.to(self.cfg.device)

            g_true.append(data.y.data.cpu())
            # print(data.y.data.cpu().numpy().tolist())
            att_label = data.edge_label.detach().cpu()
            edge_true += att_label

            # predicted_label, probs, logit, Refactor_loss, Inference_loss, edge_att = None, None, None, None, None, None
            pred_e_logit, pred_g_logit, loss_i = run_one_batch(data)
            # loss += loss_i
            edge_score.append(pred_e_logit.squeeze().detach().cpu())
            g_score.append(pred_g_logit.detach().cpu())
        g_score = torch.cat(g_score)
        edge_score = torch.cat(edge_score)
        g_pred = get_preds(g_score, self.multi_class)
        g_true = torch.cat(g_true)
        edge_pred = get_preds(edge_score, "binary")
        g_res = result_compute(g_true, g_pred, g_score, cur_mode, self.num_class)
        edge_res = result_compute(edge_true, edge_pred, edge_score, cur_mode, num_class=2)
        # if cur_mode == 'train':
        #     loss = loss / len(pbar)
        return g_res, edge_res, 0

        # result, R = result_compute(y_true, y_score, y_pred, att_true, att_score, cur_mode,multi_class=self.multi_class)



class GAT_static(nn.Module):
    def __init__(self,cfg, aux_info):
        super().__init__()
        self.cfg = cfg
        self.model = gat_model(cfg, aux_info)
        self.init_metric_dict = {'epoch': 0, 'acc': 0, 'roc_auc': 0}
        self.seed = cfg.seed
        self.use_edge_attr = cfg.shared_config.use_edge_attr
        self.gat = self.model.model

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.train.graph_lr)
        self.criterion = nn.CrossEntropyLoss()
        self.multi_class = aux_info['multi_label']

    def train(self,loaders,outdir,model_path):
        train_loss, test_loss = [], []

        best_edge_res, best_g_res = deepcopy(self.init_metric_dict), deepcopy(self.init_metric_dict)
        epoch, train_loss_epoch = 0, 0
        for epoch in  tqdm(range(self.cfg.train.epochs)):
            train_loss_epoch = 0
            for idx, data in enumerate(loaders['train']):
                data = process_edge(data, self.use_edge_attr)
                data = data.to(self.cfg.device)
                label = data.y.data.squeeze()
                # g_true += data.y.data.cpu()
                # print(data.y.data.cpu().numpy().tolist())
                att_label = data.edge_label.detach().cpu()
                # edge_true += att_label
                pred_g_logit,_ = self.gat(data)
                # predicted_label, probs, logit, Refactor_loss, Inference_loss, edge_att = None, None, None, None, None, None
                # pred_e_logit, pred_g_logit, loss_i = run_one_batch(data)
                loss_i  = self.criterion(pred_g_logit,label)
                self.optimizer.zero_grad()
                loss_i.backward()
                self.optimizer.step()
                train_loss_epoch += loss_i.detach().cpu()
                # self.model.eval()
                # pred_edge_logit = self.model.conv1.att(data.x, data.edge_index)
                # self.model.train()
            train_loss += [train_loss_epoch]

        # train_g_res, train_edge_res, _ = self.model.run_one_epoch(loaders['train'], cur_mode='train')
        # valid_g_res, valid_edge_res, _ = self.model.run_one_epoch(loaders['valid'], epoch, cur_mode='valid')
        test_g_res, test_edge_res, _ = self.model.run_one_epoch(loaders['test'],  cur_mode='test')

        if epoch >= 20 and (test_edge_res['roc_auc'] > best_edge_res[
            'roc_auc']):

            # ((valid_result['auc_att'] > metric_dict['best_valid_auc_x'])
            torch.save(self.model.state_dict(), model_path + '/' + (f'{epoch}_edge' + '.pt'))
            best_edge_epoch = {"best_edge_epoch": epoch}
            best_edge_res = test_edge_res
            best_edge_res.update(best_edge_epoch)
        if epoch >= 20 and (
                test_g_res['acc'] > best_g_res['acc']):  # ((valid_result['auc_att'] > metric_dict['best_valid_auc_x'])
            torch.save(self.model.state_dict(), model_path + '/' + (f'{epoch}_graph' + '.pt'))
            best_g_epoch = {"best_g_epoch": epoch}
            best_g_res = test_g_res
            best_g_res.update(best_g_epoch)

        logging.info(f'Seed: {self.seed}  Epoch: {epoch}')
        logging.info(f'test_g_res: {test_g_res}')
        logging.info(f'test_edge_res: {test_edge_res}')
        logging.info("")
        print(f'Seed: {self.seed}  Epoch: {epoch} Train_loss: {train_loss_epoch}')
        print(f'test_g_res: {list(test_g_res.items())[:-2]}')
        print(f'test_edge_res: {list(test_edge_res.items())[:-2]}')

        best_res = {'best_edge_res': best_edge_res,
                    'best_g_res': best_g_res}
        res_path = os.path.join(outdir, 'best_res.log')
        save_res(best_res, res_path)
        return train_loss,  best_edge_res, best_g_res

    def flips_train(self,loaders,cur_mode,r):
        epoch = self.cfg.train.epochs
        data_loader = loaders[cur_mode]
        g_logit = []
        flips_g_logit = []
        maintain_g_logit = []

        pbar = tqdm(data_loader)
        for idx, data in enumerate(pbar):
            data = process_edge(data, self.use_edge_attr)
            data = data.to(self.cfg.device)
            # build graph
            pred_g_logit,pred_edge_logit = self.model.model(data)
            pred_g_logit = pred_g_logit.detach().cpu()
            pred_e_logit = pred_edge_logit.squeeze().detach().cpu()
            g_logit.append(pred_g_logit)


            flips_data = delete_edge(data,r,pred_e_logit)
            flips_clf_logits, _= self.model.model(flips_data)
            flips_pred_g_logit = flips_clf_logits.detach().cpu()
            flips_g_logit.append(flips_pred_g_logit)

            maintain_data = retain_edge(data, r,pred_e_logit)
            maintain_clf_logits, _ = self.model.model(maintain_data)
            maintain_pred_g_logit = maintain_clf_logits.detach().cpu()
            maintain_g_logit.append( maintain_pred_g_logit)

        g_logit = torch.cat( g_logit)
        flips_g_logit = torch.cat(flips_g_logit)
        maintain_g_logit = torch.cat(maintain_g_logit)

        g_pred =  get_preds(g_logit, self.multi_class)
        flips_g_pred= get_preds(flips_g_logit,self.multi_class)
        maintain_g_pred = get_preds(maintain_g_logit,self.multi_class)


        maintain =  round(accuracy_score(y_true=g_pred, y_pred=maintain_g_pred), 5)
        decision_flips = 1- round(accuracy_score(y_true=g_pred, y_pred=flips_g_pred), 5)

        return maintain,decision_flips