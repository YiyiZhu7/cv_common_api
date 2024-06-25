# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import PIL
from PIL import Image
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [N,1,28,28] -> [N,16,28,28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # [N,16,28,28] -> [N,16,14,14]
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [N,16,14,14] -> [N,32,14,14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # [N,16,28,28] -> [N,32,7,7]
        )
        self.encoder_linear = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        self.decoder_linear = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 32 * 7 * 7),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),  # [N,32,7,7] -> [N,32,14,14]
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [N,32,14,14] -> [N,16,14,14]
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # [N,16,14,14] -> [N,16,28,28]
            nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [N,16,28,28] -> [N,1,28,28]
        )

    def forward(self, x):
        """
        :param x: [N,1,28,28]
        """
        # 1. 编码器进行特征提取
        x = self.encoder(x)  # [N, 1, 28,28] -> [N,10]
        # 2. 解码器进行解码恢复操作
        x = self.decoder(x)  # [N,10] -> [N,1,28,28]
        return x

    def encoder(self, x):
        x = self.encoder_conv(x)  # [N,1,28,28] -> [N,32,7,7]
        b, c, h, w = x.shape
        x = x.view(b, -1)  # [N,32,7,7] -> [N,32*7*7]
        x = self.encoder_linear(x)  # [N,32*7*7] -> [N,10]
        return x

    def decoder(self, x):
        x = self.decoder_linear(x)  # [N,10] -> [N,32*7*7]
        x = x.view(x.shape[0], 32, 7, 7)
        x = self.decoder_conv(x)  # [N,32,7,7] -> [N,1,28,28]
        x = x.sigmoid()  # [N,1,28,28] 取值范围变成了0~1之间
        x = 2 * x - 0.5  # 取值范围就变成了 -0.5~1.5 之间
        return x

# 由于手写数据集太多，所以这里改写使训练的数据集变少一点
class CustomLenMNIST(Dataset):
    def __init__(self, max_samples=None):
        super(CustomLenMNIST, self).__init__()
        self.max_samples = max_samples

        self.mnist = MNIST(
            root="../datas/MNIST",
            transform=transforms.Compose([
                transforms.ToTensor()
            ]),
            download=True
        )

    def __getitem__(self, item):
        return self.mnist[item]

    def __len__(self):
        if self.max_samples is None:
            return len(self.mnist)
        else:
            return int(min(len(self.mnist), self.max_samples))


def t0():
    net = AE()
    img = torch.rand(4, 1, 28, 28)
    pred_img = net(img)
    print(pred_img.shape)


def t1():
    to_img = transforms.ToPILImage()
    dataset = CustomLenMNIST(max_samples=20)
    random_index = np.random.permutation(len(dataset))
    for i in random_index:
        img_tensor, label = dataset[i]
        img: PIL.Image.Image = to_img(img_tensor)
        dir_path = f'../datas/MNIST/images/{label}'
        os.makedirs(dir_path, exist_ok=True)
        img.save(f'{dir_path}/{i}.png')


def training(total_epoch=1):
    dataset = CustomLenMNIST(max_samples=None)
    trian_dl = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AE().to(device)
    loss_fn = nn.MSELoss(reduction='mean')
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(total_epoch):
        model.train()
        for x, y in trian_dl:
            x, _ = x.to(device), y.to(device)
            x_pre = model(x)
            loss = loss_fn(x_pre, x)

            opt.zero_grad()
            loss.backward()
            opt.step()
            print(f"epoch:{epoch} loss:{loss.item():.5f}")

    model_save_path = f"./output/ae_03/model_{total_epoch}.pkl"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.cpu().eval(), model_save_path)


@torch.no_grad()
def use_encoder():
    to_tensor = transforms.ToTensor()
    model_save_path = "./output/ae_03/model.pkl"
    net: AE = torch.load(model_save_path, map_location='cpu')

    img_dir = "../datas/MNIST/images"
    label_names = os.listdir(img_dir)
    datas = {}
    for label_name in label_names:
        label_dir = os.path.join(img_dir, label_name)
        img_names = os.listdir(label_dir)
        for img_idx, img_name in enumerate(img_names):
            img_path = os.path.join(label_dir, img_name)
            img: PIL.Image.Image = Image.open(img_path)
            img = img.convert("L")
            img = to_tensor(img)[None]

            x = net.encoder(img)
            datas[f'{label_name}_{img_idx}'] = x[0].numpy()

    print(datas)
    df = pd.DataFrame(datas)
    output_dir = "./output/ae_03/encoder"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    df.corr().to_csv(os.path.join(output_dir, "data_corr.csv"))


@torch.no_grad()
def use_decoder():
    to_img = transforms.ToPILImage()
    model_save_path = "./output/ae_03/model_10.pkl"
    net: AE = torch.load(model_save_path, map_location='cpu')

    x = [
        -0.4560483, -0.5240747, 0.02532938, -0.3650791, -0.33519375,
        -0.09883122, -0.3428869, -0.3717637, -0.59968984, 0.09027517
    ]
    x = np.random.randn(10).astype(np.float32)  # 生成图像的时候，输入的是一个随机的向量
    z = net.decoder(torch.tensor([x]))[0]
    z = torch.clip(z, 0, 1)
    img = to_img(z)
    print(z.shape)
    print(img.size)
    img.show()


if __name__ == '__main__':
    # t0()
    # training(total_epoch=10)
    # t1()
    # use_encoder()
    use_decoder()
