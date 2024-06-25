from pathlib import Path

import torch
import torch.nn as nn


class IN(nn.Module):
    def __init__(self, num_features, eps=1e-8):
        super(IN, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones([1, num_features, 1, 1]))
        self.beta = nn.Parameter(torch.zeros([1, num_features, 1, 1]))

    def forward(self, x):
        """
        前向执行过程
        output = (x - μ) / σ * γ + β
        :param x: [N,C,H,W]
        :return: [N,C,H,W]
        """
        _mean = torch.mean(x, dim=(2, 3), keepdim=True)  # [N,C,1,1]
        _var = torch.var(x, dim=(2, 3), keepdim=True)  # [N,C,1,1]
        z = (x - _mean) / torch.sqrt(_var + self.eps) * self.gamma + self.beta
        return z


if __name__ == '__main__':
    torch.manual_seed(28)
    path_dir = Path("./output/models")
    path_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    net = IN(num_features=12)
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
    torch.save(net, str(path_dir / "in_model.pkl"))
    # state_dict: 获取当前模块的所有参数(Parameter + register_buffer)
    torch.save(net.state_dict(), str(path_dir / "in_params.pkl"))
    # pt结构的保存
    traced_script_module = torch.jit.trace(net.eval(), xs[0].cpu())
    traced_script_module.save('./output/models/in_model.pt')
