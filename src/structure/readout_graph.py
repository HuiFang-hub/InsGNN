# -*- coding: utf-8 -*-
# @Time    : 2023/8/15 19:51
# @Function:
import torch.nn as nn
import torch
from lib.criterion import Criterion
import dgl

# class readout_graph(nn.Module):
#     def __init__(self,cfg,aux_info,gnn_varphi_1,aggregator_1 ,aggregator_2,gnn_varphi_2):
#         super().__init__()
#         self.learn_edge_att = cfg.shared_config.learn_edge_att
#         self.learn_edge_att_m = cfg.shared_config.learn_edge_att_m
#         self.r_edge_p = cfg.r_edge_p
#         self.device = cfg.device
#         self.final_r = 0.7
#         self.decay_interval = 10
#         self.decay_r = 0.1
#         self.pred_loss_coef = cfg.model.pred_loss_coef
#         self.info_loss_coef =  cfg.model.info_loss_coef
#         self.kl = cfg.kl
#         # mask model
#         self.gnn_varphi_1  = gnn_varphi_1
#         self.aggregator_1 = aggregator_1
#         # subgraph model
#         self.gnn_varphi_2 = gnn_varphi_2
#         self.aggregator_2 = aggregator_2
#
#         self.choice_num = 10
#
#         self.criterion = Criterion(aux_info)
#
#     @staticmethod
#     def sampling(att_log_logit, training):
#         temp = 1
#         if training:
#             random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
#             random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
#             att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
#         else:
#             att_bern = (att_log_logit).sigmoid()
#         return att_bern
#
#     def forward(self, data,epoch, training):
#         # graph = dgl.batch(data.graph)
#         # m model
#         att_bern_m = self.edge_emb(data,self.gnn_varphi_1,self.aggregator_1,training,add_noise=False)
#         att_bern_Ei = self.edge_emb(data, self.gnn_varphi_2, self.aggregator_2,training, add_noise=True)
#         if self.learn_edge_att_m:
#           edge_att = (att_bern_m.detach()*att_bern_Ei).sigmoid()
#         else:
#             edge_att = att_bern_Ei
#         # edge_att = (att_bern_m.detach()*att_bern_Ei).sigmoid()
#         # edge_att = (att_bern_m * att_bern_Ei).sigmoid()
#         clf_logits = self.gnn_varphi_2(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
#         # count
#         # tmp = torch.bincount(att_m.squeeze().int())
#         # print(torch.bincount(att_m.squeeze().int()))
#
#         # modified m
#         # att_m_mod = copy.deepcopy(att_bern_m.squeeze())
#         # index_m = [i for i, j in enumerate(att_bern_m) if j == 1]
#         # index_0 = (random.sample(index_m, self.choice_num))
#         # for i in index_0:
#         #     att_m_mod[i] = 0
#         random_noise = torch.empty_like(att_bern_m).uniform_(1e-10, 1 - 1e-10)
#         att_m_mod = random_noise + att_bern_m  # add some weight
#
#         # edge_att_mod = (att_m_mod * att_bern_Ei.detach().squeeze()).unsqueeze(dim=1)
#         edge_att_mod = (att_m_mod * att_bern_Ei.detach())
#         if training:
#             self.gnn_varphi_2.eval()
#             clf_logits_mod = self.gnn_varphi_2(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr,
#                                       edge_atten=edge_att_mod)
#             self.gnn_varphi_2.train()
#         else:
#             clf_logits_mod = self.gnn_varphi_2(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr,
#                                               edge_atten=edge_att_mod)
#         if training:
#
#             # m loss
#             loss_normal = torch.sum(torch.abs(att_bern_m))
#
#             loss_kl = nn.functional.kl_div(clf_logits.softmax(dim=-1).detach().log(), clf_logits_mod.softmax(dim=-1),
#                                            reduction='batchmean')
#             loss_m = 0.0001 * loss_normal+self.kl*loss_kl
#             # self.optimizer_m.zero_grad()
#             # loss_m.backward()  # should be close
#             # self.optimizer_m.step()
#
#             # Ei loss
#             r = self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r)
#             loss_info = (att_bern_Ei * torch.log(att_bern_Ei / r + 1e-6) + (1 - att_bern_Ei) * torch.log(
#                 (1 - att_bern_Ei) / (1 - r + 1e-6) + 1e-6)).mean()
#             loss_pred = self.criterion(clf_logits, data.y)
#             loss_pred = loss_pred * self.pred_loss_coef
#             loss_info = loss_info * self.info_loss_coef
#             loss = loss_pred + loss_info
#         else:
#             loss = 0
#             loss_m = 0
#         return att_bern_Ei,clf_logits,loss,loss_m
#
#     def edge_emb(self,data,gnn_model,readout_edge,training,add_noise):
#         emb = gnn_model.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
#         E = readout_edge(emb, data.edge_index, data.batch)
#
#         if add_noise:
#             if training:
#                 random_noise = torch.empty_like(E).uniform_(1e-10, 1 - 1e-10)
#                 random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
#                 att_bern = (E + random_noise).sigmoid()
#             else:
#                 att_bern = E.sigmoid()
#         else:
#             att_bern = E.sigmoid()  #att_bern = ((E).sigmoid()>0.5).float()
#         return att_bern
#
#
#     def concat_message_function(self, edges):
#         return {'cat_feat': torch.cat([edges.src.ndata['feat'], edges.dst.ndata['feat']], dim=1)}
#
#     def get_embNet(self):
#         return self.embNet
#
#     def get_subgraph(self, graph, indices, att_log_logits):
#         sgh = att_log_logits[indices]
#         # sgh = ( att_log_logits[indices] * mask[indices]).squeeze()
#         subgraph = dgl.edge_subgraph(graph, indices, relabel_nodes=False)
#         subgraph.edata['feat'] = sgh
#         return subgraph, sgh
#
#     @staticmethod
#     def lift_node_att_to_edge_att(node_att, edge_index):
#         src_lifted_att = node_att[edge_index[0]]
#         dst_lifted_att = node_att[edge_index[1]]
#         edge_att = src_lifted_att * dst_lifted_att
#         return edge_att
#
#     @staticmethod
#     def get_r(decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
#         r = init_r - current_epoch // decay_interval * decay_r
#         if r < final_r:
#             r = final_r
#         return r



class readout_graph_2(nn.Module):
    def __init__(self,cfg,aux_info,gnn_varphi_1,aggregator_1 ,aggregator_2,gnn_varphi_2):
        super().__init__()
        self.learn_edge_att = cfg.shared_config.learn_edge_att
        self.learn_edge_att_m = cfg.shared_config.learn_edge_att_m
        # self.r_edge_p = cfg.r_edge_p
        self.device = cfg.device
        self.final_r = 0.7
        self.decay_interval = 10
        self.decay_r = 0.1
        # loss_para
        self.normal_coef = cfg.model.normal_coef
        self.kl_1_coef = cfg.model.kl_1_coef
        self.kl_2_coef = cfg.model.kl_2_coef
        self.GC_coef = cfg.model.GC_coef
        self.GC_delta_coef = cfg.model.GC_delta_coef

        # mask model
        self.gnn_varphi_1  = gnn_varphi_1
        self.aggregator_1 = aggregator_1
        # subgraph model
        self.gnn_varphi_2 = gnn_varphi_2
        self.aggregator_2 = aggregator_2

        self.choice_num = 10

        self.criterion = Criterion(aux_info)

    @staticmethod
    def sampling(att_log_logit, training):
        temp = 1
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern

    def forward(self, data,epoch, training):
        # graph = dgl.batch(data.graph)
        # m model
        M = self.edge_emb(data,self.gnn_varphi_1,self.aggregator_1,training,add_noise=False)
        edge_emb = self.edge_emb(data, self.gnn_varphi_2, self.aggregator_2,training, add_noise=True)
        # if self.learn_edge_att_m:
        #   edge_att = (M.detach()*edge_emb).sigmoid()
        # else:
        #     edge_att = edge_emb

        # edge_att = (att_bern_m.detach()*att_bern_Ei).sigmoid()
        # edge_att = (att_bern_m * att_bern_Ei).sigmoid()
        # clf_logits = self.gnn_varphi_2(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
        # count
        # tmp = torch.bincount(att_m.squeeze().int())
        # print(torch.bincount(att_m.squeeze().int()))

        # modified m
        # att_m_mod = copy.deepcopy(att_bern_m.squeeze())
        # index_m = [i for i, j in enumerate(att_bern_m) if j == 1]
        # index_0 = (random.sample(index_m, self.choice_num))
        # for i in index_0:
        #     att_m_mod[i] = 0
        random_noise = torch.empty_like(M).uniform_(1e-10, 1 - 1e-10)
        M_delta = random_noise + M  # add some weight

        # edge_att_mod = (att_m_mod * att_bern_Ei.detach().squeeze()).unsqueeze(dim=1)
        edge_M = (M * edge_emb.detach())
        edge_M_delta = (M_delta * edge_emb.detach())
        if training:
            self.gnn_varphi_2.eval()
            clf_logits_M = self.gnn_varphi_2(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr,
                                      edge_atten=edge_M)
            clf_logits_M_delta = self.gnn_varphi_2(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr,
                                               edge_atten=edge_M_delta)
            self.gnn_varphi_2.train()
        else:
            clf_logits_M = self.gnn_varphi_2(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr,
                                             edge_atten=edge_M)
            clf_logits_M_delta = self.gnn_varphi_2(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr,
                                                   edge_atten=edge_M_delta)
        if training:

            # m loss
            loss_normal = 0.0001 * torch.sum(torch.abs(M))

            loss_kl_1 = nn.functional.kl_div(
                (clf_logits_M.softmax(dim=-1) + 0.001).log(),
                clf_logits_M_delta.softmax(dim=-1),
                reduction='batchmean')
            # loss_m = 0.0001 * loss_normal+self.kl*loss_kl

            # Ei loss
            r = self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r)
            loss_kl_2 = (edge_emb * torch.log(edge_emb / r + 1e-6) + (1 - edge_emb) * torch.log(
                 (1 - edge_emb) / (1 - r + 1e-6) + 1e-6)).mean()
            loss_GC = self.criterion(clf_logits_M, data.y)
            loss_GC_delta = self.criterion(clf_logits_M_delta, data.y)
            # loss_GC = (loss_GC+loss_GC_delta) * self.pred_loss_coef
            # loss_kl = loss_kl_1 * self.info_loss_coef
            loss = self.normal_coef*loss_normal + self.kl_1_coef*loss_kl_1 +self.kl_2_coef*loss_kl_2 + self.GC_coef * loss_GC + self.GC_delta_coef * loss_GC_delta
            # loss = loss_GC +  loss_kl_2
        else:
            loss = 0
            loss_m = 0
            loss_GC, loss_kl = 0,0
        return edge_emb,clf_logits_M,loss

    def edge_emb(self,data,gnn_model,readout_edge,training,add_noise):
        emb = gnn_model.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        E = readout_edge(emb, data.edge_index, data.batch)

        if add_noise:
            if training:
                random_noise = torch.empty_like(E).uniform_(1e-10, 1 - 1e-10)
                random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
                att_bern = (E + random_noise).sigmoid()
            else:
                att_bern = E.sigmoid()
        else:
            att_bern = E.sigmoid() #att_bern = ((E).sigmoid()>0.5).float()
        return att_bern


    def concat_message_function(self, edges):
        return {'cat_feat': torch.cat([edges.src.ndata['feat'], edges.dst.ndata['feat']], dim=1)}

    def get_embNet(self):
        return self.embNet

    def get_subgraph(self, graph, indices, att_log_logits):
        sgh = att_log_logits[indices]
        # sgh = ( att_log_logits[indices] * mask[indices]).squeeze()
        subgraph = dgl.edge_subgraph(graph, indices, relabel_nodes=False)
        subgraph.edata['feat'] = sgh
        return subgraph, sgh

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def get_r(decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

class readout_graph(nn.Module):
    def __init__(self,cfg,aux_info,gnn_varphi_1,aggregator_1 ,aggregator_2,gnn_varphi_2):
        super().__init__()
        self.learn_edge_att = cfg.shared_config.learn_edge_att
        self.learn_edge_att_m = cfg.shared_config.learn_edge_att_m
        # self.r_edge_p = cfg.r_edge_p
        self.device = cfg.device
        self.final_r = 0.7
        self.decay_interval = 10
        self.decay_r = 0.1
        # loss_para
        self.normal_coef = cfg.model.normal_coef
        self.kl_1_coef = cfg.model.kl_1_coef
        self.kl_2_coef = cfg.model.kl_2_coef
        self.GC_coef = cfg.model.GC_coef
        self.GC_delta_coef = cfg.model.GC_delta_coef

        # mask model
        self.gnn_varphi_1  = gnn_varphi_1
        self.aggregator_1 = aggregator_1
        # subgraph model
        self.gnn_varphi_2 = gnn_varphi_2
        self.aggregator_2 = aggregator_2

        self.choice_num = 10

        self.criterion = Criterion(aux_info)

    @staticmethod
    def sampling(att_log_logit, training):
        temp = 1
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern

    def forward(self, data,epoch, training):
        # graph = dgl.batch(data.graph)
        # m model
        M = self.edge_emb(data,self.gnn_varphi_1,self.aggregator_1,training,add_noise=False)
        edge_emb = self.edge_emb(data, self.gnn_varphi_2, self.aggregator_2,training, add_noise=True)
        # if self.learn_edge_att_m:
        #   edge_att = (M.detach()*edge_emb).sigmoid()
        # else:
        #     edge_att = edge_emb

        # edge_att = (att_bern_m.detach()*att_bern_Ei).sigmoid()
        # edge_att = (att_bern_m * att_bern_Ei).sigmoid()
        clf_logits = self.gnn_varphi_2(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_emb)

        random_noise = torch.empty_like(M).uniform_(1e-10, 1 - 1e-10)
        M_delta = random_noise + M  # add some weight

        # edge_att_mod = (att_m_mod * att_bern_Ei.detach().squeeze()).unsqueeze(dim=1)
        edge_M = (M * edge_emb.detach())
        edge_M_delta = (M_delta * edge_emb.detach())
        if training:
            self.gnn_varphi_2.eval()
            clf_logits_M = self.gnn_varphi_2(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr,
                                      edge_atten=edge_M)
            clf_logits_M_delta = self.gnn_varphi_2(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr,
                                               edge_atten=edge_M_delta)
            self.gnn_varphi_2.train()
        else:
            clf_logits_M = self.gnn_varphi_2(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr,
                                             edge_atten=edge_M)
            clf_logits_M_delta = self.gnn_varphi_2(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr,
                                                   edge_atten=edge_M_delta)
        if training:

            # m loss
            loss_normal = torch.sum(torch.abs(M))

            loss_kl_1 =  nn.functional.kl_div(clf_logits.softmax(dim=-1).detach().log(), clf_logits_M.softmax(dim=-1),
                                           reduction='batchmean')
            r = self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r)
            loss_kl_2 = (edge_emb * torch.log(edge_emb / r + 1e-6) + (1 - edge_emb) * torch.log(
                (1 - edge_emb) / (1 - r + 1e-6) + 1e-6)).mean()

            loss_GC = self.criterion(clf_logits, data.y)
            loss_GC_delta = self.criterion(clf_logits_M_delta, data.y)

            loss_m = self.normal_coef*loss_normal + self.kl_1_coef*loss_kl_1+ self.GC_delta_coef * loss_GC_delta
            loss =self.kl_2_coef*loss_kl_2 + self.GC_coef * loss_GC

        else:
            loss = 0
            loss_m = 0
            loss_GC, loss_kl = 0,0
        return edge_emb,clf_logits_M,loss,loss_m

    def edge_emb(self,data,gnn_model,readout_edge,training,add_noise):
        emb = gnn_model.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        E = readout_edge(emb, data.edge_index, data.batch)

        if add_noise:
            if training:
                random_noise = torch.empty_like(E).uniform_(1e-10, 1 - 1e-10)
                random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
                att_bern = (E + random_noise).sigmoid()
            else:
                att_bern = E.sigmoid()
        else:
            att_bern = E.sigmoid() #att_bern = ((E).sigmoid()>0.5).float()
        return att_bern


    def concat_message_function(self, edges):
        return {'cat_feat': torch.cat([edges.src.ndata['feat'], edges.dst.ndata['feat']], dim=1)}

    def get_embNet(self):
        return self.embNet

    def get_subgraph(self, graph, indices, att_log_logits):
        sgh = att_log_logits[indices]
        # sgh = ( att_log_logits[indices] * mask[indices]).squeeze()
        subgraph = dgl.edge_subgraph(graph, indices, relabel_nodes=False)
        subgraph.edata['feat'] = sgh
        return subgraph, sgh

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def get_r(decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r