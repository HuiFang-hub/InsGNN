# -*- coding: utf-8 -*-
# @Time    : 2023/8/16 11:28
# @Function:
import os
import glob
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.metrics import accuracy_score,recall_score,roc_curve,auc,f1_score
import torch
import numpy as np
import fnmatch
def result_compute( y_true, y_pred, y_score,cur_mode,num_class):
    roc_auc, fpr, tpr = result_auc(y_true, y_score,num_class)
    acc, recall, f1 = result_acc_rec_f1(y_true, y_pred,num_class)
    recall_macro, recall_weight = 0,0
    # recall_macro, recall_weight = calculate_macro_recall(y_true, y_pred)
    if cur_mode == 'train':
        fpr, tpr = 0, 0
    results = {
        # f"avg_loss": round(ctx.loss_epoch_total / ctx.num_samples, 4),
        'roc_auc': round(roc_auc, 5),
        'acc': acc,
        'f1': f1,
        'recall': recall,
        'recall_macro': recall_macro, 'recall_weight': recall_weight,
        # f'prec_5': prec_5, f'prec_10': prec_10,
        # f'prec_20': prec_20,
        'fpr': fpr,
        'tpr': tpr
    }
    return results


def result_auc(labels, score,num_class):
    # roc_auc = roc_auc_score(y_true=labels, y_score=pred)
    if num_class>=3:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_class):
            fpr[i], tpr[i], _ = roc_curve(labels == i, score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr = np.around(fpr[0], decimals=5).tolist()
        tpr = np.around(tpr[0], decimals=5).tolist()
        roc_auc = np.around(sum( roc_auc.values()) / num_class, decimals=5).tolist()
    else:
        fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=score)
        fpr = np.around(fpr, decimals=5).tolist()
        tpr = np.around(tpr, decimals=5).tolist()
        roc_auc = round(auc(fpr, tpr),5)
    return roc_auc, fpr, tpr

def result_prec(labels, score): # time consuming
    prec_5 = round(precision_at_k(labels, score, k=5), 5)
    prec_10 = round(precision_at_k(labels, score, k=10), 5)
    prec_20 = round(precision_at_k(labels, score, k=20), 5)
    return  prec_5, prec_10, prec_20

def precision_at_k(y_true, y_pred, k=5):
    sorted_indices = np.cfgort(y_pred)[::-1]  # 按预测分数降序排列的索引
    top_k_indices = sorted_indices[:k]  # 取前k个索引
    top_k_labels = y_true[top_k_indices]  # 前k个索引对应的真实标签
    precision = round(np.sum(top_k_labels) / k ,5) # 计算精确度
    return precision

def result_acc_rec_f1(labels, pred,num_class):
    # outlier detection is a binary classification problem
    acc = round(accuracy_score(y_true=labels, y_pred=pred), 5)
    if num_class >= 3:
        recall = round(recall_score(y_true=labels, y_pred=pred, zero_division=0,average='micro'), 5)
        f1 = round(f1_score(y_true=labels, y_pred=pred,average='micro'), 5)
    else:
        recall = round(recall_score(y_true=labels, y_pred=pred,zero_division=0), 5)
        f1 = round(f1_score(y_true=labels, y_pred=pred), 5)

    return acc,recall,f1

def calculate_macro_recall(labels, predictions):
    macro_recall = round(recall_score(labels, predictions, average='macro',zero_division=0), 5)

    sample_weights = [0.3 if num == 0 else 0.7 for num in predictions]
    # 计算召回率（recall score）并考虑样本权重
    recall_weight = round(
        recall_score(labels, predictions, pos_label=1, average='weighted', sample_weight=sample_weights,zero_division=0), 5)
    return macro_recall,recall_weight

def get_preds(logits, multi_label):
    if multi_label:
        preds = (logits.sigmoid() > 0.5).float()
    elif logits.shape[1] > 1:  # multi-class
        preds = logits.argmax(dim=1).float()
    else:  # binary
        preds = (logits.sigmoid() > 0.5).float()
    return preds

def load_checkpoint(model, model_dir, model_name, map_location=None):
    checkpoint = torch.load(model_dir+'/'+ (model_name + '.pt')) #, map_location=map_location
    model.model.load_state_dict(checkpoint)
    return model

def load_checkpoint2(model, model_dir, map_location=None):
    checkpoint = torch.load(model_dir) #, map_location=map_location
    model.model.load_state_dict(checkpoint)
    return model

def delete_file(path,pattern):

    # # 使用 glob 模块查找所有匹配的文件
    # matching_files = glob.glob(os.path.join(path, pattern))
    #
    # # 遍历匹配的文件列表并删除它们
    # for file_path in matching_files:
    #     os.remove(file_path)
    #     print(f"Deleted file: {file_path}")
    for root, dirs, files in os.walk(path):
        for filename in fnmatch.filter(files, pattern):
            file_path = os.path.join(root, filename)
            # 删除文件
            os.remove(file_path)
            print(f"已删除文件: {file_path}")
