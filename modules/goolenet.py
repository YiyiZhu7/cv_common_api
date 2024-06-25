import torch
import torch.nn as nn
from torch.onnx import TrainingMode


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        """
        全局平均池化
        :param x: [N,C,H,W]
        :return: [N,C,1,1]
        """
        return torch.mean(x, dim=(2, 3), keepdim=True)


# class BasicConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         return self.relu(self.conv(x))

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(self.relu(self.bn(x)))


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: 输入通道数目, eg: 192
        :param out_channels: 各个分支的输出通道数目， eg: [[64], [96,128], [16,32], [32]]
        """
        super(Inception, self).__init__()

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels[0][0], kernel_size=1, stride=1, padding=0)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels[1][0], kernel_size=1, stride=1, padding=0),
            BasicConv2d(out_channels[1][0], out_channels[1][1], kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels[2][0], kernel_size=1, stride=1, padding=0),
            BasicConv2d(out_channels[2][0], out_channels[2][1], kernel_size=5, stride=1, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, 1, padding=1),
            BasicConv2d(in_channels, out_channels[3][0], kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        """
        inception前向过程
        :param x: [N,C,H,W]
        :return:
        """
        x1 = self.branch1(x)  # [N,C,H,W] -> [N,C1,H,W]
        x2 = self.branch2(x)  # [N,C,H,W] -> [N,C2,H,W]
        x3 = self.branch3(x)  # [N,C,H,W] -> [N,C3,H,W]
        x4 = self.branch4(x)  # [N,C,H,W] -> [N,C4,H,W]
        x = torch.concat([x1, x2, x3, x4], dim=1)  # [N,C1+C2+C3+C4,H,W]
        return x


class GoogLeNet(nn.Module):
    def __init__(self, num_classes, add_aux_stage=False):
        super(GoogLeNet, self).__init__()
        self.stage1 = nn.Sequential(
            BasicConv2d(3, 64, 7, 2, 3),
            nn.MaxPool2d(3, 2, padding=1),
            BasicConv2d(64, 64, 1, 1, 0),
            BasicConv2d(64, 192, 3, 1, 1),
            nn.MaxPool2d(3, 2, padding=1),
            Inception(192, [[64], [96, 128], [16, 32], [32]]),  # inception3a
            Inception(256, [[128], [128, 192], [32, 96], [64]]),  # inception3b
            nn.MaxPool2d(3, 2, padding=1),
            Inception(480, [[192], [96, 208], [16, 48], [64]])  # inception4a
        )
        self.stage2 = nn.Sequential(
            Inception(512, [[160], [112, 224], [24, 64], [64]]),  # inception4b
            Inception(512, [[128], [128, 256], [24, 64], [64]]),  # inception4c
            Inception(512, [[112], [144, 288], [32, 64], [64]])  # inception4d
        )
        self.stage3 = nn.Sequential(
            Inception(528, [[256], [160, 320], [32, 128], [128]]),  # inception4e
            nn.MaxPool2d(3, 2, padding=1),
            Inception(832, [[256], [160, 320], [32, 128], [128]]),  # inception5a
            Inception(832, [[384], [192, 384], [48, 128], [128]]),  # inception5b
            GlobalAvgPool2d()
        )
        self.classify = nn.Conv2d(1024, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=0)
        if add_aux_stage:
            self.aux_stage1 = nn.Sequential(
                nn.MaxPool2d(5, 3, padding=0),
                nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), padding=0),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(output_size=(2, 2)),
                nn.Flatten(1),
                nn.Linear(4096, 2048),
                nn.Dropout(p=0.4),
                nn.ReLU(),
                nn.Linear(2048, num_classes)
            )

            self.aux_stage2 = nn.Sequential(
                nn.MaxPool2d(5, 3, padding=0),
                nn.Conv2d(528, 1024, kernel_size=(1, 1), stride=(1, 1), padding=0),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(output_size=(2, 2)),
                nn.Flatten(1),
                nn.Linear(4096, 2048),
                nn.Dropout(p=0.4),
                nn.ReLU(),
                nn.Linear(2048, num_classes)
            )
        else:
            self.aux_stage1 = None
            self.aux_stage2 = None

    def forward(self, x):
        """
        前向过程
        :param x: [N,3,H,W]
        :return:
        """
        z1 = self.stage1(x)  # [N,3,H,W] -> [N,512,H1,W1]
        z2 = self.stage2(z1)  # [N,512,H1,W1] -> [N,528,H2,W2]
        z3 = self.stage3(z2)  # [N,528,H2,W2] -> [N,1024,1,1]

        # 三个决策分支的输出
        # scores3 = self.classify(z3)[:, :, 0, 0]  # [N,1024,1,1] -> [N,num_classes,1,1] -> [N,num_classes]
        scores3 = torch.squeeze(self.classify(z3))  # [N,1024,1,1] -> [N,num_classes,1,1] -> [N,num_classes]
        if self.aux_stage1 is not None:
            scores1 = self.aux_stage1(z1)
            scores2 = self.aux_stage2(z2)
            return scores1, scores2, scores3
        else:
            return scores3


def t1():
    # inception = Inception(192, [[64], [96, 128], [16, 32], [32]])
    # print(inception)
    # _x = torch.rand(4, 192, 100, 100)
    # _r = inception(_x)
    # print(_r.shape)

    # net = GoogLeNet(num_classes=4)
    # _x = torch.randn(2, 3, 224, 224)
    # _r = net(_x)
    # print(_r)
    # print(_r.shape)

    net = GoogLeNet(num_classes=4, add_aux_stage=True)
    loss_fn = nn.CrossEntropyLoss()
    _x = torch.randn(2, 3, 224, 224)  # 模拟的输入图像原始数据
    _y = torch.tensor([0, 3], dtype=torch.long)  # 模拟的真实标签id
    _r1, _r2, _r3 = net(_x)  # 获取三个决策分支的预测置信度，可以用来和实际标签一起构建损失
    _loss1 = loss_fn(_r1, _y)
    _loss2 = loss_fn(_r2, _y)
    _loss3 = loss_fn(_r3, _y)
    loss = _loss1 + _loss2 + _loss3
    print(_r1)
    print(_r2)
    print(_r3)
    print(_r3.shape)
    print(loss)

    # NOTE: 这里只是为了让大家可视化看结构，实际情况中，如果转换为pt或者onnx的时候，记住一定需要将aux分支删除
    traced_script_module = torch.jit.trace(net.eval(), _x)
    traced_script_module.save('./output/models/googlenet_aux.pt')

    # 模型持久化
    torch.save(net, './output/models/googlenet.pkl')


def t2():
    # 参数加载
    net1 = torch.load('./output/models/googlenet.pkl', map_location='cpu')

    net2 = GoogLeNet(num_classes=4, add_aux_stage=False)
    # missing_keys: 表示net2中有部分参数没有恢复
    # unexpected_keys: 表示net2中没有这部分参数，但是入参的字典中传入了该参数
    # net1.state_dict(): 返回的是一个dict，key是参数的名称字符串，value是参数tensor对象
    missing_keys, unexpected_keys = net2.load_state_dict(net1.state_dict(), strict=False)
    if len(missing_keys) > 0:
        raise ValueError(f"网络有部分参数没有恢复:{missing_keys}")
    print(unexpected_keys)

    _x = torch.randn(2, 3, 224, 224)  # 模拟的输入图像原始数据
    traced_script_module = torch.jit.trace(net2.eval(), _x)
    traced_script_module.save('./output/models/googlenet.pt')

    torch.onnx.export(
        model=net2.eval().cpu(),  # 给定模型对象
        args=_x,  # 给定模型forward的输出参数
        f='./output/models/googlenet_dynamic.onnx',  # 输出文件名称
        training=TrainingMode.EVAL,  # 训练还是eval阶段
        do_constant_folding=True,
        input_names=['images'],  # 给定输入的tensor名称列表
        output_names=['scores'],  # 给定输出的tensor名称列表
        opset_version=12,
        dynamic_axes={
            'images': {
                0: 'n',
                2: 'h',
                3: 'w'
            },
            'scores': {
                0: 'n'
            }
        }  # 给定是否是动态结构
    )


if __name__ == '__main__':
    t1()
    t2()
