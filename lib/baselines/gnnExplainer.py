import logging
import os
import dgl.function as fn
import torch
import torch.nn as nn
# from dgl.nn import AvgPooling, GNNExplainer
from torch_geometric.nn import global_add_pool
from lib.data_loader.process_data import process_edge
from tqdm import tqdm
from copy import deepcopy
from lib.model_loader.get_model import get_model
import dgl
from log.save_results import save_res,save_model
from lib.criterion import Criterion
from lib.result_comp import get_preds,result_compute
from sklearn.metrics import accuracy_score
from lib.data_loader.process_data import process_edge, delete_edge, retain_edge
from lib.explain import Explainer,GNNExplainer
# class Model(nn.Module):
#     def __init__(self, in_feats, out_feats):
#         super(Model, self).__init__()
#         self.linear = nn.Linear(in_feats, out_feats)
#         self.pool = AvgPooling()
#         self.fc_out = nn.Sequential(nn.Linear(out_feats, out_feats))
#     def forward(self, graph, feat,batch=None,eweight=None):
#         # feat = graph.ndata['feat']
#         with graph.local_scope():
#             feat = self.linear(feat)
#             graph.ndata['h'] = feat
#             if eweight is None:
#                 graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
#             else:
#                 graph.edata['w'] = eweight
#                 graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
#             if batch !=None:
#                 return self.fc_out(global_add_pool(graph.ndata['h'],batch))
#             else:
#                 return self.pool(graph, graph.ndata['h'])

class gnnexp(nn.Module):
    def __init__(self, cfg, aux_info):
        super().__init__()
        self.node_feat_dim, self.num_class = aux_info['node_feat_dim'], aux_info['num_class']
        self.cfg = cfg
        self.gnn = get_model(aux_info, cfg)

        self.explainer = Explainer(
        model=self.gnn,
        algorithm= GNNExplainer(aux_info,cfg.device,epochs=cfg.train.epochs),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='graph',
            return_type='probs',
        ),
    )
        # self.model_dir = model_dir
        self.use_edge_attr = cfg.shared_config.use_edge_attr
        self.multi_class = aux_info['multi_label']

    def eval_one_batch(self, data, epoch=None):
        pred_g_logit, pred_edge_logit = None, None
        # train_loss = self.explainer.algorithm.train(epoch, self.gnn, data.x, data.edge_index, data.batch, target=data.y)
        if epoch >= self.cfg.train.epochs - 1:
            explanation = self.explainer(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr,target = data.y)
            pred_g_logit = explanation.prediction
            pred_edge_logit = explanation.edge_mask
        return pred_edge_logit, pred_g_logit, 0

    def train_one_batch(self, data, epoch=None):
        pred_g_logit, pred_edge_logit = None, None
        # output = self.gnn(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        if epoch >= self.cfg.train.epochs - 1:
            explanation = self.explainer(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr,target = data.y)
            pred_g_logit = explanation.prediction
            pred_edge_logit = explanation.edge_mask

        # graph = dgl.batch(data.graph)
        # pred_g_logit = self.gnn(graph, graph.ndata['feat'])
        # label = data.y.squeeze()
        # train_loss = self.criterion(pred_g_logit, label.long())
        # self.optimizer.zero_grad()
        # train_loss.backward()
        # self.optimizer.step()
        # feat_att, pred_edge_logit = self.explainer.explain_graph(graph, graph.ndata['feat'], batch=data.batch)
        return pred_edge_logit, pred_g_logit, 0

    def run_one_epoch(self, data_loader, cur_mode,epoch):
        run_one_batch = self.train_one_batch if cur_mode == 'train' else self.eval_one_batch
        loss = 0
        edge_true, edge_score = [], []
        g_true, g_score = [], []
        pbar = tqdm(data_loader)
        for idx, data in enumerate(pbar):
            data = process_edge(data, self.use_edge_attr)
            data = data.to(self.cfg.device)
            # output = self.gnn(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
            g_true.append(data.y.data.cpu())
            att_label = data.edge_label.detach().cpu()
            edge_true += att_label
            pred_e_logit, pred_g_logit, _ = run_one_batch(data, epoch)
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
        #     loss = loss.detach().cpu() / len(pbar)
        return g_res, edge_res, loss




class GNNExplainer_static(nn.Module):
    def __init__(self, cfg,aux_info):
        super(GNNExplainer_static, self).__init__()
        # self.node_feat_dim, self.num_class = aux_info['node_feat_dim'], aux_info['num_class']
        self.cfg = cfg
        self.aux_info = aux_info
        self.init_metric_dict = {'epoch': 0, 'acc': 0, 'roc_auc': 0}
        self.seed = cfg.seed
        self.model = gnnexp(cfg, aux_info)

        self.criterion = Criterion(aux_info)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.train.graph_lr)
        self.use_edge_attr = cfg.shared_config.use_edge_attr
        self.multi_class = aux_info['multi_label']

    def train(self, loaders, outdir, model_path):
        train_loss, test_loss = [], []

        best_edge_res, best_g_res = deepcopy(self.init_metric_dict), deepcopy(self.init_metric_dict)
        # train model
        train_loss_epoch = 0

        epoch = self.cfg.train.epochs

        # self.model = gnnexp(self.cfg, self.aux_info, self.gnn)
        train_g_res, train_edge_res,_ = self.model.run_one_epoch(loaders['train'], cur_mode='train',epoch = epoch)
        # valid_g_res, valid_edge_res,_ = self.run_one_epoch(loaders['valid'], epoch, cur_mode='valid')
        test_g_res, test_edge_res, _ = self.model.run_one_epoch(loaders['test'], cur_mode='test', epoch=epoch)

        if epoch >= 20 and (test_edge_res['roc_auc'] > best_edge_res[
            'roc_auc']):  # ((valid_result['auc_att'] > metric_dict['best_valid_auc_x'])
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


    def flips_train(self, loaders, cur_mode, r):
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

            explanation = self.model.explainer(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr,target=data.y)
            pred_g_logit = explanation.prediction
            pred_e_logit = explanation.edge_mask
            pred_g_logit = pred_g_logit.detach().cpu()
            pred_e_logit = pred_e_logit.squeeze().detach().cpu()
            g_logit.append(pred_g_logit)

            flips_data = delete_edge(data, r, pred_e_logit)
            explanation = self.model.explainer(flips_data.x, flips_data.edge_index, batch=flips_data.batch,
                                               edge_attr=flips_data.edge_attr,target=flips_data.y)
            flips_clf_logits = explanation.prediction
            flips_pred_g_logit = flips_clf_logits.detach().cpu()
            flips_g_logit.append(flips_pred_g_logit)

            maintain_data = retain_edge(data, r, pred_e_logit)
            explanation = self.model.explainer(maintain_data.x, maintain_data.edge_index, batch=maintain_data.batch,
                                               edge_attr=maintain_data.edge_attr,target=maintain_data.y)
            maintain_clf_logits = explanation.prediction
            maintain_pred_g_logit = maintain_clf_logits.detach().cpu()
            maintain_g_logit.append(maintain_pred_g_logit)

        g_logit = torch.cat(g_logit)
        flips_g_logit = torch.cat(flips_g_logit)
        maintain_g_logit = torch.cat(maintain_g_logit)

        g_pred = get_preds(g_logit, self.multi_class)
        flips_g_pred = get_preds(flips_g_logit, self.multi_class)
        maintain_g_pred = get_preds(maintain_g_logit, self.multi_class)

        maintain = round(accuracy_score(y_true=g_pred, y_pred=maintain_g_pred), 5)
        decision_flips = 1 - round(accuracy_score(y_true=g_pred, y_pred=flips_g_pred), 5)

        return maintain, decision_flips

