import torch
from .model import Model
import argparse

args = argparse.Namespace(
    model='Transolver',
    n_hidden=128,
    n_heads=8,
    n_layers=8,
    mlp_ratio=2,
    fun_dim=2,
    space_dim=2,
    out_dim=1,
    geotype='structured_2D',
    shapelist=[256, 4],
    slice_num=64,
    unified_pos=0,
    ref=8,
    dropout=0.0,
    act='gelu',
    time_input=False,
    normalize=False,
)

# 加载模型
model = Model(args).cuda()
model.load_state_dict(torch.load("/home/amiya/code/py/Python/BY/piern_airfoil/src/piern_airfoil/transolver/checkpoints/airfoil_Transolver.pt"))
model.eval()

# 准备输入 [B, N, 2] 坐标 + [B, N, 2] 条件
x = torch.randn(1, 1024, 2).cuda()   # 坐标
fx = torch.randn(1, 1024, 2).cuda()  # 物理条件

# 推理
with torch.no_grad():
    pred = model(x, fx)  # → [1, 1024, 1]
print(pred.shape)  # 输出预测结果的形状