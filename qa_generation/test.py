import torch
import torch.nn as nn

# 假设有 3 个类，logits 形状为 (batch_size=3, num_classes=3)
logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3], [1.5, 0.5, 2.0]])

# 标签，其中第二个样本的标签为 ignore_index (-100)
labels = torch.tensor([0, -100, 2])

# 定义 CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# 计算损失
loss = criterion(logits, labels)

print(f"Loss: {loss}")