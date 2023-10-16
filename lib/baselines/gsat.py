import sys
sys.path.append('../src')
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from lib.data_loader.process_data import delete_edge, retain_edge
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from torch_sparse import transpose
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import subgraph, is_undirected
from torch_geometric.loader import DataLoader
from ogb.graphproppred import Evaluator
from copy import deepcopy
from lib.result_comp import get_preds
from lib.model_loader.get_model import get_model
from src.structure.readout_edge import readout_edge
from lib.result_comp import result_compute
import logging
import os
from log.save_results import save_res
from lib.criterion import Criterion
from lib.data_loader.process_data import process_edge
class gsat_model(nn.Module):
    def __init__(self, cfg, aux_info):
        super().__init__()
        self.device = cfg.device
        self.seed = cfg.seed
        self.clf = get_model(aux_info, cfg)
        self.extractor = readout_edge(cfg.model.hidden_size, cfg.shared_config).to(cfg.device)
        self.optimizer = torch.optim.Adam(list(self.extractor.parameters()) + list(self.clf.parameters()),
                                          lr=cfg.train.graph_lr, weight_decay=cfg.train.weight_decay)
        self.dataset_name = cfg.data.data_name
        self.random_state = cfg.seed
        self.model_dir = cfg.model.model_save_path
        self.use_edge_attr = cfg.shared_config.use_edge_attr
        self.learn_edge_att = cfg.shared_config.learn_edge_att
        # self.k = shared_config['precision_k']
        self.num_viz_samples = cfg.shared_config.num_viz_samples
        self.viz_interval = cfg.shared_config.viz_interval
        self.viz_norm_att = cfg.shared_config.viz_norm_att

        self.epochs = cfg.train.epochs
        self.pred_loss_coef = cfg.model.GC_coef
        self.info_loss_coef = cfg.model.kl_2_coef
        self.fix_r = cfg.model.fix_r
        self.final_r = cfg.model.final_r
        self.decay_interval = cfg.model.decay_interval
        self.decay_r = cfg.model.decay_r
        self.init_r = 0.9

        self.multi_class = aux_info['multi_label']
        self.num_class = aux_info['num_class']
        self.criterion = Criterion(aux_info)
        # self.criterion = nn.BCEWithLogitsLoss(weight=None, reduce=False)

    def __loss__(self, att, clf_logits, clf_labels, epoch):
        pred_loss = self.criterion(clf_logits, clf_labels)

        r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r,
                                                     init_r=self.init_r)
        info_loss = (att * torch.log(att / r + 1e-6) + (1 - att) * torch.log((1 - att) / (1 - r + 1e-6) + 1e-6)).mean()

        pred_loss = pred_loss * self.pred_loss_coef
        info_loss = info_loss * self.info_loss_coef
        loss = pred_loss + info_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item()}
        return loss, loss_dict

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    def forward_pass(self, data, epoch, training):
        emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        att_log_logits = self.extractor(emb, data.edge_index, data.batch)
        att = self.sampling(att_log_logits, epoch, training)

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                nodesize = data.x.shape[0]
                edge_att = (att + transpose(data.edge_index, att, nodesize, nodesize, coalesced=False)[1]) / 2
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)

        clf_logits = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
        loss, loss_dict = self.__loss__(att, clf_logits, data.y, epoch)
        return edge_att, loss, clf_logits

    @torch.no_grad()
    def eval_one_batch(self, data, epoch):
        self.extractor.eval()
        self.clf.eval()
        att, loss,  clf_logits = self.forward_pass(data, epoch, training=False)
        return att.data.cpu().reshape(-1),  clf_logits.data.cpu(),loss

    def train_one_batch(self, data, epoch):
        self.extractor.train()
        self.clf.train()
        att, loss,  clf_logits = self.forward_pass(data, epoch, training=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return att.data.cpu().reshape(-1),clf_logits.data.cpu(), loss

    def run_one_epoch(self, data_loader, epoch, cur_mode):
        loader_len = len(data_loader)
        run_one_batch = self.train_one_batch if cur_mode == 'train' else self.eval_one_batch
        phase = 'test ' if cur_mode == 'test' else cur_mode  # align tqdm desc bar
        g_true, g_score = [], []
        edge_true, edge_score = [], []
        loss = 0

        all_loss_dict = {}
        # all_exp_labels, all_att, all_clf_labels, all_clf_logits, all_precision_at_k = ([] for i in range(5))
        pbar = tqdm(data_loader)
        for idx, data in enumerate(pbar):
            data = process_edge(data, self.use_edge_attr)
            pred_e_logit, pred_g_logit,loss_i = run_one_batch(data.to(self.device), epoch)
            edge_label = data.edge_label.cpu().numpy().tolist()
            edge_true += edge_label
            loss += loss_i
            edge_score.append(pred_e_logit)
            # all_precision_at_k.extend(precision_at_k)
            g_true.append(data.y.data.cpu()), g_score.append(pred_g_logit)
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

    def sampling(self, att_log_logits, epoch, training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern


class GSAT(nn.Module):
    def __init__(self, cfg, aux_info):
        super().__init__()
        self.cfg = cfg
        self.model = gsat_model(cfg, aux_info)
        self.seed = cfg.seed
        self.epochs = cfg.train.epochs
        self.init_metric_dict = {'epoch': 0, 'acc': 0,
                                 'roc_auc': 0}
        self.use_edge_attr = cfg.shared_config.use_edge_attr
        self.multi_class = aux_info['multi_label']

    def train(self,loaders,outdir,model_path):
        # viz_set = self.get_viz_idx(test_set, self.dataset_name)
        train_loss, test_loss = [], []
        best_edge_epoch, best_g_epoch = 0, 0
        best_edge_res, best_g_res = deepcopy(self.init_metric_dict), deepcopy(self.init_metric_dict)
        for epoch in range(self.epochs):
            train_g_res, train_edge_res, train_loss_epoch = self.model.run_one_epoch(loaders['train'],epoch, cur_mode='train')
            # valid_g_res, valid_edge_res, _ = self.model.run_one_epoch(loaders['valid'], epoch, cur_mode='valid')
            test_g_res, test_edge_res, _ = self.model.run_one_epoch(loaders['test'], epoch, cur_mode='test')
            train_loss += [train_loss_epoch]

            if epoch >= 20 and (test_edge_res['roc_auc'] > best_edge_res['roc_auc']):  # ((valid_result['auc_att'] > metric_dict['best_valid_auc_x'])
                torch.save(self.model.state_dict(), model_path + '/' + (f'{epoch}_edge' + '.pt'))
                best_edge_epoch = {"best_edge_epoch": epoch}
                best_edge_res = test_edge_res
                best_edge_res.update(best_edge_epoch)
            if epoch >= 20 and (test_g_res['acc'] > best_g_res['acc']):  # ((valid_result['auc_att'] > metric_dict['best_valid_auc_x'])
                torch.save(self.model.state_dict(), model_path + '/' + (f'{epoch}_graph' + '.pt'))
                best_g_epoch = {"best_g_epoch": epoch}
                best_g_res = test_g_res
                best_g_res.update(best_g_epoch)
            logging.info(f'Seed: {self.seed}  Epoch: {epoch}')
            logging.info(f'test_g_res: {test_g_res}')
            logging.info(f'test_edge_res: {test_edge_res}')
            logging.info("")
            print(f'Seed: {self.seed}  Epoch: {epoch} Train_Loss:{train_loss_epoch}')
            print(f'test_g_res: {list(test_g_res.items())[:-2]}')
            print(f'test_edge_res: {list(test_edge_res.items())[:-2]}')
        best_res = {'best_edge_res': best_edge_res,
                    'best_g_res': best_g_res}
        res_path = os.path.join(outdir, 'best_res.log')
        save_res(best_res, res_path)
        return train_loss,  best_edge_res,best_g_res

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
            att_bern_Ei, _,  clf_logits = self.model.forward_pass(data, epoch, training=True)
            pred_e_logit = att_bern_Ei.cpu().reshape(-1)
            pred_g_logit = clf_logits.data.cpu()
            g_logit.append(pred_g_logit)

            flips_data = delete_edge(data, r, pred_e_logit)
            _, _, flips_clf_logits= self.model.forward_pass(flips_data, epoch, training=False)
            flips_pred_g_logit = flips_clf_logits.data.cpu()
            flips_g_logit.append(flips_pred_g_logit)

            maintain_data = retain_edge(data, r, pred_e_logit)
            _,_, maintain_clf_logits = self.model.forward_pass(maintain_data, epoch, training=False)
            maintain_pred_g_logit = maintain_clf_logits.data.cpu()
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

    def log_epoch(self, epoch, phase, loss_dict, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
        desc = f'[Seed {self.random_state}, Epoch: {epoch}]: gsat_{phase}........., ' if batch else f'[Seed {self.random_state}, Epoch: {epoch}]: gsat_{phase} finished, '
        for k, v in loss_dict.items():
            # if not batch:
                # self.writer.add_scalar(f'gsat_{phase}/{k}', v, epoch)
            desc += f'{k}: {v:.3f}, '

        eval_desc, att_auroc, precision, clf_acc, clf_roc = self.get_eval_score(epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch)
        desc += eval_desc
        return desc, att_auroc, precision, clf_acc, clf_roc, loss_dict['pred']

    def get_eval_score(self, epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
        clf_preds = get_preds(clf_logits, self.multi_label)
        clf_acc = 0 if self.multi_label else (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]

        if batch:
            return f'clf_acc: {clf_acc:.3f}', None, None, None, None

        precision_at_k = np.mean(precision_at_k)
        clf_roc = 0
        if 'ogb' in self.dataset_name:
            evaluator = Evaluator(name='-'.join(self.dataset_name.split('_')))
            clf_roc = evaluator.eval({'y_pred': clf_logits, 'y_true': clf_labels})['rocauc']

        att_auroc, bkg_att_weights, signal_att_weights = 0, att, att
        if np.unique(exp_labels).shape[0] > 1:
            att_auroc = roc_auc_score(exp_labels, att)
            bkg_att_weights = att[exp_labels == 0]
            signal_att_weights = att[exp_labels == 1]

        desc = f'clf_acc: {clf_acc:.3f}, clf_roc: {clf_roc:.3f}, ' + \
               f'att_roc: {att_auroc:.3f}, att_prec@{self.k}: {precision_at_k:.3f}'
        return desc, att_auroc, precision_at_k, clf_acc, clf_roc



def save_checkpoint(model, model_dir, model_name):
    torch.save({'model_state_dict': model.state_dict()}, model_dir / (model_name + '.pt'))

