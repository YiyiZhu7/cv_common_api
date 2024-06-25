import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx import TrainingMode


class VggBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n, use_11=False):
        super(VggBlock, self).__init__()
        layers = []
        for i in range(n):
            if use_11 and (i == n - 1):
                kernel_size = (1, 1)
                padding = 0
            else:
                kernel_size = (3, 3)
                padding = 1
            conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=(1, 1), padding=padding),
                nn.ReLU()
            )
            in_channel = out_channel
            layers.append(conv)
        layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AdaptiveAvgPool2dModule(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dModule, self).__init__()
        self.k = output_size

    def forward(self, x):
        k = self.k
        n, c, h, w = x.shape
        hk = int(h / k)
        if h % k != 0:
            hk += 1
        wk = int(w / k)
        if w % k != 0:
            wk += 1
        ph = hk * k - h  # 需要填充大小
        pw = wk * k - w  # 需要填充大小
        x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        x = x.reshape(n, c, k, hk, k, wk)
        x = torch.permute(x, dims=(0, 1, 2, 4, 3, 5))
        x = torch.mean(x, dim=(4, 5))
        return x


class VggNet(nn.Module):
    def __init__(self, features, num_classes, classify_input_channel):
        super(VggNet, self).__init__()
        self.num_classes = num_classes

        self.features = features  # 特征提取网络
        # self.pooling = nn.AdaptiveAvgPool2d(output_size=7)
        self.pooling = AdaptiveAvgPool2dModule(output_size=7)
        self.classify = nn.Sequential(
            nn.Linear(in_features=7 * 7 * classify_input_channel, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )

    def forward(self, images):
        """
        网络的前向执行过程

        :param images: [N,3,H,W] 原始图像信息
        :return: [N,num_classes] 预测类别置信度
        """
        z = self.features(images)  # [N,3,H,W] -> [N,classify_input_channel,?,?]
        z = self.pooling(z)  # [N,classify_input_channel,?,?] --> [N,classify_input_channel,7,7]
        z = z.flatten(1)  # [N,classify_input_channel,7,7] -> [N,classify_input_channel*7*7]
        return self.classify(z)


class Vgg16CNet(nn.Module):
    def __init__(self, num_classes):
        super(Vgg16CNet, self).__init__()

        features = nn.Sequential(
            VggBlock(3, 64, 2),
            VggBlock(64, 128, 2),
            VggBlock(128, 256, 3, use_11=True),
            VggBlock(256, 512, 3, use_11=True),
            VggBlock(512, 512, 3, use_11=True)
        )

        self.vgg = VggNet(
            features=features,
            num_classes=num_classes,
            classify_input_channel=512
        )

    def forward(self, images):
        return self.vgg(images)


class Vgg16Net(nn.Module):
    def __init__(self, num_classes):
        super(Vgg16Net, self).__init__()

        features = nn.Sequential(
            VggBlock(3, 64, 2),
            VggBlock(64, 128, 2),
            VggBlock(128, 256, 3),
            VggBlock(256, 512, 3),
            VggBlock(512, 512, 3)
        )

        self.vgg = VggNet(
            features=features,
            num_classes=num_classes,
            classify_input_channel=512
        )

    def forward(self, images):
        return self.vgg(images)


class Vgg19Net(nn.Module):
    def __init__(self, num_classes):
        super(Vgg19Net, self).__init__()

        features = nn.Sequential(
            VggBlock(3, 64, 2),
            VggBlock(64, 128, 2),
            VggBlock(128, 256, 4),
            VggBlock(256, 512, 4),
            VggBlock(512, 512, 4)
        )

        self.vgg = VggNet(
            features=features,
            num_classes=num_classes,
            classify_input_channel=512
        )

    def forward(self, images):
        return self.vgg(images)


class VggLabelNet(nn.Module):
    def __init__(self, vgg):
        super(VggLabelNet, self).__init__()
        self.vgg = vgg
        self.id2name = {
            0: '小狗',
            1: '小猫',
            2: '牛',
            3: '羊'
        }

    def forward(self, images):
        scores = self.vgg(images)  # [N,C,H, W] --> [N,num_classes]
        pred_indexes = torch.argmax(scores, dim=1)  # [N,num_classes] --> [N]
        pred_indexes = pred_indexes.detach().numpy()
        result = []
        for idx in pred_indexes:
            result.append(self.id2name[idx])
        return pred_indexes


if __name__ == '__main__':
    vgg16 = Vgg16Net(num_classes=4)
    vgg_label = VggLabelNet(vgg16)
    print(vgg_label)
    example = torch.rand(4, 3, 352, 224)
    r = vgg_label(example)  # 先调用__call__方法，然后再调用forward方法
    print(r)

    torch.onnx.export(
        model=vgg16.eval().cpu(),  # 给定模型对象
        args=example,  # 给定模型forward的输出参数
        f='./output/models/vgg16_dynamic.onnx',  # 输出文件名称
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
    # traced_script_module = torch.jit.trace(vgg16, example)
    # traced_script_module.save('./output/models/vgg16.pt')
