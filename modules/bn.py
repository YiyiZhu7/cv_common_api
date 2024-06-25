from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class BN(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-8):
        """
        初始化方法
        :param num_features: 特征属性的数量，也就是通道数目C
        """
        super(BN, self).__init__()

        self.momentum = momentum
        self.eps = eps

        # register_buffer: 将属性当成parameter进行处理，唯一的区别就是不参与反向传播的梯度求解
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_var', torch.zeros(1, num_features, 1, 1))
        self.running_mean: Optional[Tensor]
        self.running_var: Optional[Tensor]

        self.gamma = nn.Parameter(torch.ones([1, num_features, 1, 1]))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        """
        前向过程
        output = (x - μ) / σ * γ + β
        :param x: [N,C,H,W]
        :return: [N,C,H,W]
        """
        if self.training:
            # 训练阶段 --> 使用当前批次的数据
            _mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)  # [1,C,1,1]
            _var = torch.var(x, dim=(0, 2, 3), keepdim=True)  # [1,C,1,1]
            # 将训练过程中的均值和方差保存下来-方便推理的时候使用 --> 滑动平均
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * _mean
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * _var
        else:
            # 推理阶段 --> 使用的是训练过程中的累计数据
            _mean = self.running_mean
            _var = self.running_var
        z = (x - _mean) / torch.sqrt(_var + self.eps) * self.gamma + self.beta
        return z


if __name__ == '__main__':
    torch.manual_seed(28)
    path_dir = Path("./output/models")
    path_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    bn = BN(num_features=12)
    bn.to(device)  # 只针对子模块或者参数进行转换

    # 模拟训练过程
    bn.train()
    xs = [torch.randn(8, 12, 32, 32).to(device) for _ in range(10)]
    for _x in xs:
        bn(_x)

    print(bn.running_mean.view(-1))
    print(bn.running_var.view(-1))

    # 模拟推理过程
    bn.eval()
    _r = bn(xs[0])
    print(_r.shape)

    bn = bn.cpu()
    # 模拟模型保存
    torch.save(bn, str(path_dir / "bn_model.pkl"))
    # state_dict: 获取当前模块的所有参数(Parameter + register_buffer)
    torch.save(bn.state_dict(), str(path_dir / "bn_params.pkl"))
    # pt结构的保存
    traced_script_module = torch.jit.trace(bn.eval(), xs[0].cpu())
    traced_script_module.save('./output/models/bn_model.pt')

    # 模拟模型恢复
    bn_model = torch.load(str(path_dir / "bn_model.pkl"), map_location='cpu')
    bn_params = torch.load(str(path_dir / "bn_params.pkl"), map_location='cpu')
    print(len(bn_params))
