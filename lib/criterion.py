# -*- coding: utf-8 -*-
# @Time    : 2023/8/15 19:54
# @Function:
import torch.nn as nn
import torch.nn.functional as F
class Criterion(nn.Module):
    def __init__(self, aux_info):
        super(Criterion, self).__init__()
        self.num_class = aux_info['num_class']
        self.multi_label = aux_info['multi_label']
        # print(f'[INFO] Using multi_label: {self.multi_label}')

    def forward(self, logits, targets):
        if self.num_class == 2 and not self.multi_label:
            loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        elif self.num_class > 2 and not self.multi_label:
            loss = F.cross_entropy(logits, targets.long())
        else:
            is_labeled = targets == targets  # mask for labeled data
            loss = F.binary_cross_entropy_with_logits(logits[is_labeled], targets[is_labeled].float())
        return loss