# -*- coding: utf-8 -*-
# @Time    : 2023/8/15 19:00
# @Function:
import os

import torch.nn as nn
import torch
from tqdm import tqdm
from copy import deepcopy
import logging
from lib.model_loader.get_model import get_model
from src.structure.readout_edge import readout_edge
from src.structure.readout_graph import readout_graph
from lib.data_loader.process_data import process_edge, delete_edge, retain_edge, InsGNN_subgraph
from lib.result_comp import get_preds, result_compute, delete_file
from log.save_results import save_res
from lib.criterion import Criterion
from sklearn.metrics import accuracy_score
class subg(nn.Module):
    def __init__(self, cfg, aux_info):
        super().__init__()
        self.cfg = cfg

        # other parameters
        # self.pred_loss_coef = cfg.model.pred_loss_coef
        # self.info_loss_coef = cfg.model.info_loss_coef
        self.lr, self.wd = cfg.train.graph_lr, cfg.train.weight_decay  # self.lr = method_config['lr']
        self.use_edge_attr = cfg.shared_config.use_edge_attr

        self.model_dir = cfg.model.model_save_path
        self.multi_class = aux_info['multi_label']
        self.num_class = aux_info['num_class']

        # m model
        self.gnn_varphi_1 = get_model(aux_info, cfg)
        self.aggregator_1 = readout_edge(cfg.model.hidden_size, cfg.shared_config).to(cfg.device)
        # E_i model
        self.gnn_varphi_2 = get_model(aux_info, cfg)
        self.aggregator_2 = readout_edge(cfg.model.hidden_size, cfg.shared_config).to(cfg.device)

        # main model
        self.Inference = readout_graph(cfg, aux_info,
                                       self.gnn_varphi_1, self.aggregator_1,
                                       self.aggregator_2, self.gnn_varphi_2).to(cfg.device)
        # optimizer & criterion
        self.optimizer_m = torch.optim.Adam(list(self.gnn_varphi_1.parameters()) + list(self.aggregator_1.parameters()),
                                            lr=self.lr,
                                            weight_decay=self.wd)
        self.optimizer_E = torch.optim.Adam(list(self.gnn_varphi_2.parameters()) + list(self.aggregator_2.parameters()),
                                            lr=self.lr,
                                            weight_decay=self.wd)
        self.optimizer = torch.optim.Adam(list(self.gnn_varphi_1.parameters()) + list(self.aggregator_1.parameters()) +
                                          list(self.gnn_varphi_2.parameters()) + list(self.aggregator_2.parameters())
                                          # + list(self.Inference.parameters())
                                          ,
                                          lr=self.lr,
                                          weight_decay=self.wd)

        self.adversarial_loss = nn.CrossEntropyLoss()
        # self.binary_criterion =  F.binary_cross_entropy_with_logits()
        # self.bce_criterion = nn.BCEWithLogitsLoss(weight=None, reduce=False)
        self.criterion = Criterion(aux_info)

    def train_one_batch(self, data, epoch):
        data.y = data.y.type(torch.LongTensor).to(self.cfg.device)
        # self.Discriminator.train()
        self.gnn_varphi_1.train()
        self.aggregator_1.train()
        self.gnn_varphi_2.train()
        self.aggregator_2.train()
        # ---------------------
        #  Train Inference
        # ---------------------
        att_bern_Ei, clf_logits, loss, loss_m = self.Inference(data, epoch, training=True)
        self.optimizer_m.zero_grad()
        loss_m.backward()  # should be close
        self.optimizer_m.step()

        self.optimizer_E.zero_grad()
        loss.backward()  # should be close
        self.optimizer_E.step()
        # loss = loss_GC+loss_kl
        # self.optimizer.zero_grad()
        # loss.backward()  # should be close
        # self.optimizer.step()

        pred_e_logit = att_bern_Ei.detach().cpu()
        pred_g_logit = clf_logits.data.cpu()

        return pred_e_logit, pred_g_logit, loss

    def eval_one_batch(self, data, epoch):
        self.gnn_varphi_1.eval()
        self.aggregator_1.eval()
        self.gnn_varphi_2.eval()
        self.aggregator_2.eval()
        self.Inference.eval()

        att_bern_Ei, clf_logits, loss, _ = self.Inference(data, epoch, training=False)

        pred_e_logit = att_bern_Ei.detach().cpu()
        pred_g_logit = clf_logits.data.cpu()

        return pred_e_logit, pred_g_logit, loss

    def run_one_epoch(self, data_loader, epoch, cur_mode):
        run_one_batch = self.train_one_batch if cur_mode == 'train' else self.eval_one_batch
        g_true, g_score = [], []
        edge_true, edge_score = [], []
        loss = 0
        pbar = tqdm(data_loader)
        for idx, data in enumerate(pbar):
            data = process_edge(data, self.use_edge_attr)
            data = data.to(self.cfg.device)
            # label = data.y.squeeze()
            g_true.append(data.y.data.cpu())
            # print(data.y.data.cpu().numpy().tolist())
            edge_label = data.edge_label.detach().cpu().numpy().tolist()
            edge_true += edge_label
            # build graph
            pred_e_logit, pred_g_logit, loss_i = run_one_batch(data, epoch)
            loss += loss_i
            # y_pred += y_p
            edge_score.append(pred_e_logit)
            g_score.append(pred_g_logit)
        g_score = torch.cat(g_score)
        edge_score = torch.cat(edge_score)
        g_pred = get_preds(g_score, self.multi_class)
        g_true = torch.cat(g_true)
        edge_pred = get_preds(edge_score, "binary")
        g_res = result_compute(g_true, g_pred, g_score, cur_mode, self.num_class)
        edge_res = result_compute(edge_true, edge_pred, edge_score, cur_mode, num_class=2)
        if cur_mode == 'train':
            loss = loss.detach().cpu() / len(pbar)
        return g_res, edge_res, loss


class InsGNN(nn.Module):
    def __init__(self, cfg, aux_info):
        super().__init__()
        self.model = subg(cfg, aux_info)
        self.cfg = cfg
        self.init_metric_dict = {'epoch': 0, 'acc': 0, 'roc_auc': 0}
        self.seed = cfg.seed
        self.use_edge_attr = cfg.shared_config.use_edge_attr
        self.multi_class = aux_info['multi_label']

    def train(self,loaders,outdir,model_path):
        train_loss,test_loss = [],[]
        best_edge_epoch,best_g_epoch = 0,0
        best_edge_res,best_g_res = deepcopy(self.init_metric_dict),deepcopy(self.init_metric_dict)
        for epoch in range(self.cfg.train.epochs):
            train_g_res, train_edge_res, train_loss_epoch = self.model.run_one_epoch(loaders['train'],epoch, cur_mode='train')
            # valid_g_res, valid_edge_res, _ = self.model.run_one_epoch(loaders['valid'], epoch, cur_mode='valid')
            test_g_res, test_edge_res, _ = self.model.run_one_epoch(loaders['test'], epoch, cur_mode='test')
            train_loss += [train_loss_epoch]
            # test_loss += [test_loss_epoch]
            if epoch>= 20 and (test_edge_res['roc_auc'] > best_edge_res['roc_auc']) : #((valid_result['auc_att'] > metric_dict['best_valid_auc_x'])
                pattern = '*_edge.pt'
                delete_file(model_path, pattern)
                torch.save(self.model.state_dict(),  os.path.join(model_path,f'{epoch}_edge.pt'))
                best_edge_epoch = {"best_edge_epoch":epoch}
                best_edge_res = test_edge_res
                best_edge_res.update(best_edge_epoch)
            if epoch>= 20 and (test_g_res['acc'] > best_g_res['acc']) : #((valid_result['auc_att'] > metric_dict['best_valid_auc_x'])
                pattern = '*_graph.pt'
                delete_file(model_path, pattern)
                torch.save(self.model.state_dict(), os.path.join(model_path, f'{epoch}_graph.pt'))
                best_g_epoch = {"best_g_epoch":epoch}
                best_g_res = test_g_res
                best_g_res.update(best_g_epoch)

            logging.info(f'Seed: {self.seed}  Epoch: {epoch}')
            logging.info(f'test_g_res: {test_g_res}')
            logging.info(f'test_edge_res: {test_edge_res}')
            logging.info("")
            print(f'Seed: {self.seed}  Epoch: {epoch} Train_loss: {train_loss_epoch}')
            print(f'test_g_res: {list(test_g_res.items())[:-2]}')
            print(f'test_edge_res: {list(test_edge_res.items())[:-2]}')

        best_res = {'best_edge_res':best_edge_res,
                  'best_g_res': best_g_res}
        res_path = os.path.join(outdir, 'best_res.log')
        save_res(best_res,res_path)
        return  train_loss,  best_edge_res,best_g_res

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
            att_bern_Ei, clf_logits, _, _ = self.model.Inference(data, epoch, training=False)
            pred_g_logit = clf_logits.data.cpu()
            pred_e_logit = att_bern_Ei.detach().cpu()
            g_logit.append(pred_g_logit)

            flips_data, maintain_data = InsGNN_subgraph(data, pred_e_logit)

            # flips_data = delete_edge(data,r,pred_e_logit)
            _, flips_clf_logits, _, _ = self.model.Inference(flips_data, epoch, training=False)
            flips_pred_g_logit = flips_clf_logits.data.cpu()
            flips_g_logit.append(flips_pred_g_logit)

            # maintain_data = retain_edge(data, r,pred_e_logit)
            _, maintain_clf_logits, _, _ = self.model.Inference(maintain_data, epoch, training=False)
            maintain_pred_g_logit = maintain_clf_logits.data.cpu()
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






    @staticmethod
    def get_r(decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r
