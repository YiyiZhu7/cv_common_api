from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# noinspection DuplicatedCode,PyAttributeOutsideInit,PyMethodMayBeStatic
class SN(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-8):
        """
        初始化方法
        :param num_features: 特征属性的数量，也就是通道数目C
        """
        super(SN, self).__init__()

        self.momentum = momentum
        self.eps = eps

        # register_buffer: 将属性当成parameter进行处理，唯一的区别就是不参与反向传播的梯度求解
        self.register_buffer('running_bn_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_bn_var', torch.zeros(1, num_features, 1, 1))
        self.running_bn_mean: Optional[Tensor]
        self.running_bn_var: Optional[Tensor]

        self.gamma = nn.Parameter(torch.ones([1, num_features, 1, 1]))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

        self.w = nn.Parameter(torch.ones([3]))

    def get_bn(self, x):
        if self.training:
            # 训练阶段 --> 使用当前批次的数据
            _bn_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)  # [1,C,1,1]
            _bn_var = torch.var(x, dim=(0, 2, 3), keepdim=True)  # [1,C,1,1]
            # 将训练过程中的均值和方差保存下来-方便推理的时候使用 --> 滑动平均
            self.running_bn_mean = self.momentum * self.running_bn_mean + (1.0 - self.momentum) * _bn_mean
            self.running_bn_var = self.momentum * self.running_bn_var + (1.0 - self.momentum) * _bn_var
        else:
            # 推理阶段 --> 使用的是训练过程中的累计数据
            _bn_mean = self.running_bn_mean
            _bn_var = self.running_bn_var
        return _bn_mean, _bn_var

    def get_ln(self, x):
        _mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)  # [N,1,1,1]
        _var = torch.var(x, dim=(1, 2, 3), keepdim=True)  # [N,1,1,1]
        return _mean, _var

    def get_in(self, x):
        _mean = torch.mean(x, dim=(2, 3), keepdim=True)  # [N,C,1,1]
        _var = torch.var(x, dim=(2, 3), keepdim=True)  # [N,C,1,1]
        return _mean, _var

    def forward(self, x):
        """
        前向过程
        output = (x - μ) / σ * γ + β
        :param x: [N,C,H,W]
        :return: [N,C,H,W]
        """
        # 均值和方差
        _bn_mean, _bn_var = self.get_bn(x)
        _ln_mean, _ln_var = self.get_ln(x)
        _in_mean, _in_var = self.get_in(x)
        # 权重
        w = torch.softmax(self.w, dim=0)
        bn_w, ln_w, in_w = w[0], w[1], w[2]
        # 合并
        _mean = _bn_mean * bn_w + _ln_mean * ln_w + _in_mean * in_w
        _var = _bn_var * bn_w + _ln_var * ln_w + _in_var * in_w
        # 转换
        z = (x - _mean) / torch.sqrt(_var + self.eps) * self.gamma + self.beta
        return z


if __name__ == '__main__':
    torch.manual_seed(28)
    path_dir = Path("./output/models")
    path_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    net = SN(num_features=12)
    net.to(device)  # 只针对子模块或者参数进行转换

    # 模拟训练过程
    net.train()
    xs = [torch.randn(8, 12, 32, 32).to(device) for _ in range(10)]
    for _x in xs:
        net(_x)

    # 模拟推理过程
    net.eval()
    _r = net(xs[0])
    print(_r.shape)

    net = net.cpu()
    # 模拟模型保存
    torch.save(net, str(path_dir / "sn_model.pkl"))
    # state_dict: 获取当前模块的所有参数(Parameter + register_buffer)
    torch.save(net.state_dict(), str(path_dir / "sn_params.pkl"))
    # pt结构的保存
    traced_script_module = torch.jit.trace(net.eval(), xs[0].cpu())
    traced_script_module.save('./output/models/sn_model.pt')
