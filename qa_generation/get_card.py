import torch

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建随机矩阵
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)

# 持续进行矩阵乘法计算
while True:
    c = torch.matmul(a, b)  # 不断做矩阵乘法
