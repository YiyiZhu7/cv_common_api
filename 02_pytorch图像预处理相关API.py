import torch
from PIL import Image
from torchvision import transforms



"""

transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像或者numpy对象形式的图像数据转换为tensor对象
    transforms.ToPILImage(),  # 将tensor对象转换为PIL对象
    transforms.Normalize(mean=[], std=[]),  # 基于给定的每个通道的均值和标准差进行标准化操作, 要求输入必须是tensor
    transforms.Resize(size=100),  # 图像大小resize
    transforms.CenterCrop(size=20),  # 从图像中心位置进行图像的截取
    transforms.Pad(padding=5),  # 对图像进行H和W的填充
    transforms.RandomApply(transforms=None, p=0.5),  # 安装给定概率执行给定的transforms
    transforms.RandomOrder(transforms=[]),  # 随机顺序执行转换操作
    transforms.RandomChoice(transforms=[]),  # 随机选择一个执行转换操作
    transforms.RandomCrop(size=10),  # 随机剪切图像
    transforms.RandomHorizontalFlip(p=0.5),  # 按照给定概率随机的水平翻转给定图像
    transforms.RandomVerticalFlip(p=0.5),  # 按照给定概率随机的垂直翻转给定图像
    transforms.RandomPerspective(),  # 按照给定概率随机透视转换
    transforms.RandomResizedCrop(size=10),  # 按照给定参数进行随机剪切并resize成给定大小
    transforms.FiveCrop(size=10),  # 截取左上、右上、左下、右下、中间的五个图像
    transforms.TenCrop(size=10, vertical_flip=False),  # 截取原始图像的五个子图像 + 水平翻转后的五个子图像
    transforms.LinearTransformation(transformation_matrix=None, mean_vector=None),  # 线性变化，基于给定的参数针对图像进行白化操作
    transforms.ColorJitter(),  # 随机更改图像的亮度、对比度、饱和度以及色调
    transforms.RandomRotation(degrees=30),  # 随机旋转图像
    transforms.RandomAffine(degrees=30),  # 仿射变换
    transforms.Grayscale(),  # RGB图像转换为灰度图像
    transforms.RandomGrayscale(p=0.2),  # 按照给定概率随机将RGB图像转换为灰度图像
    transforms.RandomErasing(),  # 按照给定概率随机对图像进行矩形区域擦拭掩盖
    transforms.GaussianBlur(kernel_size=3),  # 对图像做高斯blur滤波
    transforms.RandomInvert(),  # 反转图像
    transforms.RandomPosterize(bits=3),  # 针对每个channel保留几个bits来设置颜色，默认是256种颜色也就是8位，要求入参必须是uint8
    transforms.ConvertImageDtype(torch.uint8),  # 图像转换类型
    transforms.RandomSolarize(threshold=0.2),  # 基于给定概率随机将高于threshold的像素点进行反转
    transforms.RandomAdjustSharpness(sharpness_factor=2),  # 基于概率随机调整清晰度/锐度
    transforms.RandomAutocontrast(),  # 基于概率对图像进行差异化转换其实就是区间缩放(x - min) / (max - min) * (255 or 1.0)
    transforms.RandomEqualize()  # 基于给定概率均衡直方图
])

"""


@torch.no_grad()
def t1():
    import numpy as np
    img = Image.open(r"../datas/小狗.png")
    img = img.convert("RGB")

    ts0 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.uint8),
            transforms.RandomPosterize(bits=3, p=1.0)
        ]
    )
    print(np.unique(np.array(img)))
    print(np.unique(ts0(img).data.numpy()))
    transforms.ToPILImage()(ts0(img)).show()

    ts = transforms.Compose(
        transforms=[
            transforms.ToTensor(),  # PIL转换为tensor对象
            transforms.RandomOrder(
                transforms=[
                    transforms.RandomPerspective(),
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomAffine(degrees=30),
                    transforms.RandomErasing(),
                    transforms.RandomAutocontrast(),
                    transforms.Resize((244, 244)),
                ]
            ),
            # transforms.CenterCrop(size=(200, 100)),  # 基于中心点进行图像的剪切
            transforms.RandomCrop(size=(150, 150)),
            # transforms.Resize(size=(400, 300))
            # transforms.Normalize(mean=[0.2, 0.7, 0.5], std=[0.22, 0.17, 0.26])
        ]
    )
    ts2 = transforms.Compose(
        transforms=[
            transforms.ToPILImage()  # tensor对象转换为PIL对象
        ]
    )
    tensor = ts(img)
    # print(tensor)
    print(tensor.size())
    img2 = ts2(tensor)
    img2.show()

    img3 = ts2(ts(img))
    img3.show()


if __name__ == '__main__':
    t1()
