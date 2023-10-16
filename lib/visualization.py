# -*- coding: utf-8 -*-
# @Time    : 2023/8/17 11:05
# @Function:
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, to_networkx
import dgl
from tqdm import tqdm
import torch
from rdkit import Chem
import networkx as nx
from torch_geometric.data import Data
from mne.io import RawArray, read_raw_edf
from lib.data_loader.process_data import process_edge
import seaborn as sns
from ExplainerParser import Parser
def plot_loss(epochs, train_loss,path):
    # 创建一个新的图形
    plt.figure(figsize=(10, 6))

    # 绘制训练损失和测试损失的折线图
    plt.plot(epochs, train_loss, marker='o', label='Train Loss')
    # plt.plot(epochs, test_loss, marker='o', label='Test Loss')

    # 添加标题和标签
    plt.title('Train Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # 添加图例
    plt.legend()
    plt.savefig(path)
    # plt.show()

def plot_auc(fpr,tpr,path):
    """
    绘制 AUC 曲线
    参数:
    fpr (list): 假正率列表
    tpr (list): 真正率列表
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(path)

def eval_one_batch(data, model, criterion, optimizer=None):
    assert optimizer is None
    model.eval()
    logits = model(data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
    loss = criterion(logits, data.y)
    return loss.item(), logits.data.cpu(), data.y.data.cpu()

def get_viz_idx(test_set, dataset_name, num_viz_samples):
    y_dist = test_set.data.y.numpy().reshape(-1)
    num_nodes = np.array([each.x.shape[0] for each in test_set])
    classes = np.unique(y_dist)
    res = []
    for each_class in classes:
        tag = 'class_' + str(each_class)
        if dataset_name == 'Graph-SST2':
            condi = (y_dist == each_class) * (num_nodes > 5) * (num_nodes < 10)  # in case too short or too long
            candidate_set = np.nonzero(condi)[0]
        else:
            candidate_set = np.nonzero(y_dist == each_class)[0]
        idx = np.random.choice(candidate_set, num_viz_samples, replace=False)
        res.append((idx, tag))

    if dataset_name == 'mutag':
        for each_class in classes:
            tag = 'class_' + str(each_class)
            candidate_set = np.nonzero(y_dist == each_class)[0]
            idx = np.random.choice(candidate_set, num_viz_samples, replace=False)
            res.append((idx, tag))
    return res

def visualize_results(model, all_viz_set, test_set, num_viz_samples, dataset_name, use_edge_attr,path,cfg):
    figsize = 10
    fig, axes = plt.subplots(len(all_viz_set), num_viz_samples, figsize=(figsize*num_viz_samples, figsize*len(all_viz_set)*0.8))
    model = model.to(cfg.device)
    # test_set = test_set.to(cfg.device)
    for class_idx, (idx, tag) in enumerate(all_viz_set):
        viz_set = test_set[idx]
        data = next(iter(DataLoader(viz_set, batch_size=len(idx), shuffle=False)))
        data = process_edge(data, use_edge_attr)
        data = data.to(cfg.device)

        label = data.y.squeeze()
        label = label.to(cfg.device)
        # y_true += data.y.data.cpu().numpy().tolist()
        # print(data.y.data.cpu().numpy().tolist())
        att_label = data.edge_label
        # att_true.append(att_label)
        u, v = data.edge_index[0], data.edge_index[1]
        g = dgl.graph((u, v))
        g.ndata['feat'] = data.x.float()
        graph = g.to(cfg.device)
        # if cfg.framework == 'GAN_DGNN' or cfg.framework == 'GAN_DGNN+':
        #     _,_,batch_att,_= model.eval_one_batch(graph,data,label,epoch=500)
        # elif cfg.framework == 'GAN_DGNN3'or cfg.framework == 'EastGNN4':
        #     _,_,batch_att,_ = model.eval_one_batch(data, label, epoch=500)
        # elif cfg.framework == 'GAST':
        #     batch_att, _, _ = eval_one_batch(model, data, epoch=500)
        # elif  cfg.framework == 'EastGNN5':
        #     _,batch_att, _ = model.eval_one_batch( data, label,epoch=500)
        # elif cfg.framework == 'EastGNN6' or cfg.framework == 'EastGNN7' or cfg.framework == 'InsGNN':
        #     batch_att, _,_ = model.eval_one_batch(data, epoch=500)
        # else:
        #     _, _, batch_att, _ = model.eval_one_batch(graph, label,data.batch )
        # if cfg.framework == 'InsGNN':
        #     batch_att, _,_  = model.eval_one_batch(data,epoch=500)
        # elif cfg.framework == 'GAST':
        #     batch_att, _, _ = model.eval_one_batch(data, epoch=500)
        batch_att, _, _ = model.model.eval_one_batch(data, epoch=500)


        batch_att = torch.tensor(batch_att)
        for i in tqdm(range(len(viz_set))):
            mol_type, coor = None, None
            if dataset_name == 'mutag':
                node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
                mol_type = {k: node_dict[v.item()] for k, v in enumerate(viz_set[i].node_type)}
            elif dataset_name == 'Graph-SST2':
                mol_type = {k: v for k, v in enumerate(viz_set[i].sentence_tokens)}
                num_nodes = data.x.shape[0]
                x = np.linspace(0, 1, num_nodes)
                y = np.ones_like(x)
                coor = np.stack([x, y], axis=1)
            elif dataset_name == 'ogbg_molhiv':
                element_idxs = {k: int(v+1) for k, v in enumerate(viz_set[i].x[:, 0])}
                mol_type = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v)) for k, v in element_idxs.items()}
            elif dataset_name == 'mnist':
                raise NotImplementedError

            node_subset = data.batch == i
            _, edge_mask = subgraph(node_subset, data.edge_index, edge_attr=batch_att)

            node_label = viz_set[i].node_label.reshape(-1) if viz_set[i].get('node_label', None) is not None else torch.zeros(viz_set[i].x.shape[0])
            visualize_a_graph(viz_set[i].edge_index, edge_mask, node_label, dataset_name, axes[class_idx, i], norm=True, mol_type=mol_type, coor=coor)
            # axes[class_idx, i].axis('off')
        # fig.tight_layout()

    each_plot_len = 1/len(viz_set)
    for num in range(1, len(viz_set)):
        line = plt.Line2D((each_plot_len*num, each_plot_len*num), (0, 1), color="gray", linewidth=1, linestyle='dashed', dashes=(5, 10))
        fig.add_artist(line)

    each_plot_width = 1/len(all_viz_set)
    for num in range(1, len(all_viz_set)):
        line = plt.Line2D((0, 1), (each_plot_width*num, each_plot_width*num), color="gray", linestyle='dashed', dashes=(5, 10))
        fig.add_artist(line)

    plt.savefig(path)

def visualize_a_graph(edge_index, edge_att, node_label, dataset_name, ax, coor=None, norm=False, mol_type=None, nodesize=300):
    if norm:  # for better visualization
        edge_att = edge_att**10
        edge_att = (edge_att - edge_att.min()) / (edge_att.max() - edge_att.min() + 1e-6)

    if mol_type is None or dataset_name == 'Graph-SST2':
        atom_colors = {0: '#E49D1C', 1: '#FF5357', 2: '#a1c569', 3: '#69c5ba'}
        node_colors = [None for _ in range(node_label.shape[0])]
        for y_idx in range(node_label.shape[0]):
            node_colors[y_idx] = atom_colors[node_label[y_idx].int().tolist()]
    else:
        node_color = ['#29A329', 'lime', '#F0EA00',  'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
        element_idxs = {k: Chem.PeriodicTable.GetAtomicNumber(Chem.GetPeriodicTable(), v) for k, v in mol_type.items()}
        node_colors = [node_color[(v - 1) % len(node_color)] for k, v in element_idxs.items()]

    data = Data(edge_index=edge_index, att=edge_att, y=node_label, num_nodes=node_label.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])

    # calculate Graph positions
    if coor is None:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = {idx: each.tolist() for idx, each in enumerate(coor)}

    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->" if dataset_name == 'Graph-SST2' else '-',
                lw=max(data['att'], 0) * 3,
                alpha=max(data['att'], 0),  # alpha control transparency
                color='black',  # color control color
                shrinkA=np.sqrt(nodesize) / 2.0 + 1,
                shrinkB=np.sqrt(nodesize) / 2.0 + 1,
                connectionstyle='arc3,rad=0.4' if dataset_name == 'Graph-SST2' else 'arc3'
            ))

    if mol_type is not None:
        nx.draw_networkx_labels(G, pos, mol_type, ax=ax)

    if dataset_name != 'Graph-SST2':
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=nodesize, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax, connectionstyle='arc3,rad=0.4')

def plot_line_bond(line_df, fig_path,order,legend_name):
    # mpl.rcParams['text.usetex'] = True
    # mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
    sns.set_theme(style="darkgrid")
    # hue_order = ['0.1', '0.01', '0.001']
    line_colors = ['#FFD0E9', '#B9191A', '#DBE7CA','#99BADF', '#99CDCE', '#999ACD']
    # line_colors = ['#A95465', '#A97C81','#A6DADB', '#FAB378', '#137355', '#5E5851', ]
    color_palette = dict(zip(order, line_colors))
    ax = sns.lineplot(x="fpr", y="tpr",err_style = "band", hue="method_name", hue_order=order, data=line_df, palette=color_palette)
    # ax = sns.lineplot(x="fpr", y="tpr", hue="method_name", hue_order=hue_order, data=line_df)
    ax.legend(title=legend_name)
    # lines = ax.get_lines()
    # for line in lines:
    #     if line.get_label() == "FedGAD":
    #         line.set_linewidth(5.5)

    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random guess')
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.subplots_adjust(right=0.7, top=0.7)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()

def plot_heat_map(df,fig_path,axis_param):
    sns.set_theme(style="white")
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    maxv = df.max().max()
    minv =df.min().min()
    center = df.stack().median()
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(df, cmap=cmap,vmin=minv, vmax= maxv, center=center,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    xylabel = []
    for name in axis_param:
        if name == 'normal_coef':
            label = '$\lambda_0$'
        elif name == 'kl_1_coef':
            label = '$\lambda_1$'
        else:
            label = '$\lambda_2$'
        xylabel.append(label)
    plt.xlabel(f'{xylabel[0]}', fontsize=20)
    plt.ylabel(f'{xylabel[1]}', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(fig_path)

def plot_box(df,x_name,y_name,hue_name,fig_path,order):

    line_colors = ['#B9191A','#FFD0E9',  '#DBE7CA','#99BADF', '#99CDCE', '#999ACD']
    # line_colors = ['#A95465', '#A97C81','#A6DADB', '#FAB378', '#137355', '#5E5851']
    color_palette = dict(zip(order, line_colors))
    ax = sns.boxplot(x=x_name, y=y_name,
                hue=hue_name, palette=color_palette,
                data=df)
    # sns.despine(offset=10, trim=True)
    ax.legend(title=None)
    # 设置 x 轴和 y 轴标签的字体大小
    ax.set_xlabel(x_name, fontsize=20)
    ax.set_ylabel(y_name, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.clf()


def plot_ts_curve(data_path, ts, te):
    channels_name = ['FP1-F7', 'P7-O1', 'F3-C3', 'C3-P3', 'FP2-F8', 'P8-O2', 'F4-C4', 'C4-P4', 'T7-FT9', 'FT10-T8']
    recDf = read_raw_edf(data_path).to_data_frame()
    fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(10, 6))
    for col, ax in zip(channels_name, axes):  # 跳过第一列，因为它是时间戳
        ax.plot(recDf[col].iloc[0:ts], color='#78A040')
        ax.plot(recDf[col].iloc[ts:te], color='r')
        ax.plot(recDf[col].iloc[te:], color='#78A040')
        ax.get_yaxis().set_visible(False)  # 隐藏纵轴坐标
        ax.tick_params(axis='x', labelsize=16)
    # 移除子图间的间距
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    # 显示图形
    plt.show()
    print('test')

if __name__ == '__main__':
    args = Parser(description='Explainer').args
    args.device = 5
    # device
    is_cuda = not args.disable_cuda and torch.cuda.is_available()
    if is_cuda:
        args.device = torch.device("cuda:" + str(args.device))
        torch.cuda.manual_seed_all(seed=args.seed)
    else:
        args.device = torch.device("cpu")
    data_path = '../data/chbmit/chbmit_60/raw/chb01_03.edf'  # female 14
    ts = 2996 * 256  # '13:43:04'
    te = 3036 * 256
    plot_ts_curve(data_path, ts, te)
