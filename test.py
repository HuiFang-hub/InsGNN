import torch

# 假设 logits 是一个包含模型输出的张量
logits = torch.tensor([-0.7, 0.7, 0.6, 0.8, 0.4])

# 使用 sigmoid 函数计算概率
probabilities = logits.sigmoid()

# 找到满足条件的索引
indices = (probabilities > 0.5).nonzero().squeeze()

# 输出满足条件的索引
print(indices)