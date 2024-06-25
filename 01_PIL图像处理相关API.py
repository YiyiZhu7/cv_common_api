import base64
import collections
import copy
import io

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


def t1():
    img: Image.Image = Image.open(r"../datas/小狗.png")
    # img = img.convert("L")  # RGB转换为灰度图像
    # img.show()
    print(type(img))
    img.save("d.bmp")
    w, h = img.size
    print(img.format, img.size, img.mode)
    mode = img.mode
    # img.show('动物')

    # Numpy之间可以无缝转换
    # Pillow图像转换成numpy数组后，格式为:[H,W,C]
    img_arr = np.array(img)
    if len(img_arr.shape) == 2:
        img_arr = img_arr[:, :, None]
    print(img_arr.shape)
    print(img_arr[:2, :2, :])

    # 既然这里是一个numpy数组，所以可以对里面的数字做任何向做的处理
    img_arr1 = copy.deepcopy(img_arr)
    img_arr1[:100, :100, :3] = 0
    if mode == 'L':
        img_arr1 = img_arr1[:, :, 0]
    img1 = Image.fromarray(img_arr1, mode)
    img1.save("img1.png")

    # 随机mask掩盖一部分图像区域
    for i in range(5):
        img_arr2 = copy.deepcopy(img_arr)
        s = np.random.randint(50, min(w, h) // 2)
        hi = np.random.randint(low=0, high=h - s)
        wi = np.random.randint(low=0, high=w - s)
        img_arr2[hi:hi + s, wi:wi + s, :3] = 0
        if mode == 'L':
            img_arr2 = img_arr2[:, :, 0]
        img2 = Image.fromarray(img_arr2, mode)
        img2.save(f"img2_{i}.png")


def t2():
    img: Image.Image = Image.open("../datas/小狗.png")
    img = img.convert("RGB").convert("RGB")
    img.show('rgb原始图像')
    # 1. 将图像转换为灰度图像
    """
    def convert(self, mode=None, matrix=None, dither=None,
                    palette=WEB, colors=256)
        mode=None：指定转换为什么通道的数据。 仅支持："L", "RGB" and "CMYK"
    """
    # img是RGB图像
    img1 = img.convert("L")
    img1.show()
    # img1.save("e.png")

    # 2. 灰度图像转换为黑白图像(二值化图像的操作)
    img2 = img1.point(lambda i: 255 if i > 128 else 0)
    img2.show()

    # 3. 大小缩放
    """
    resample: 给定缩放的方式
    """
    # img3 = img.resize((1024, 1024), resample=Image.NEAREST)
    img3 = img.resize((300, 50), resample=Image.NEAREST)
    img3.show()

    # 4. 图像的旋转
    """
    def rotate(self, angle, resample=NEAREST, expand=0, center=None,
                   translate=None)
        angle: 角度, 如果是正数，那就是逆时针旋转，如果是一个负数，那就顺时针旋转
        expand=0：是否做填充，0表示不做，图像大小不变；True表示做，图像大小会发生变化
    """
    # TODO: 需要自己改一下源码。(有的版本，可以给定具体的填充颜色，有的版本不支持)
    img41 = img.rotate(angle=20, expand=False, fillcolor=(255, 255, 255))
    img42 = img.rotate(angle=20, expand=True, fillcolor=(255, 255, 255))
    img41.show()
    img42.show()

    # 5. 图像的翻转(左右翻转、上下翻转、旋转90度...)
    img5 = img.transpose(Image.FLIP_TOP_BOTTOM)
    img5.show()

    # 6. 剪切
    # box为：(left, upper, right, lower)
    _img = Image.open(r"../datas/小猫.jpg")
    box = (300, 100, 900, 300)
    img6 = _img.crop(box)
    img6.show()

    # 7. 图像分裂、合并
    if np.array(img6).shape[-1] == 3:
        r, g, b = img6.split()
    elif np.array(img6).shape[-1] == 4:
        r, g, b, _ = img6.split()
    else:
        raise ValueError()
    r = r.point(lambda i: i + 100)
    img7 = Image.merge('RGB', (g, b, r))  # 把g当作r，b当作g，r到工作b
    # img7 = Image.merge('RGB', (r, g, b))
    img7.show()

    # 8. 图像的粘贴（将img6粘贴到img上，位置信息为box）
    img6_2 = img.crop((200, 200, 800, 400))
    img6_2 = Image.blend(img6_2, img6, alpha=0.5)  # 将两个size、channel完全相同的图像进行虚化的合并
    img.paste(img6_2, (200, 200, 800, 400))
    img.show()

    # 9. 最基本的图像的像素操作
    # 第一个值为宽度(x)、第二个值为高度y的坐标值(从左往右，从上往下)
    # 获取像素
    a = img.getpixel((250, 300))
    print(a)
    # 设置像素值
    for i in range(0, 100):
        for j in range(250, 350):
            img.putpixel((i, j), (0, 0, 0))
    img.show()


def t3():
    # 读取图像
    img: Image.Image = Image.open('../datas/小狗.png')
    img = img.convert("RGB")

    # 1. 使用ImageFilter/滤波器做数据增强
    img1 = img.copy()
    for i in range(10):
        img1 = img1.filter(ImageFilter.GaussianBlur)
    img1.show()

    # 2. 使用像素点做数据增强
    img2 = img.point(lambda i: i * 2)
    img2.show()

    # 3. 基于分割的调整
    r, g, b = img.split()
    mask = g.point(lambda i: i < 100 and 255)
    out_r = r.point(lambda i: i * 0.5)
    r.paste(out_r, None, mask)
    img3 = Image.merge("RGB", (r, g, b))
    img3.show()

    # 构建一个数据增强的对象
    # Contrast: 对比度调整；Color：颜色的调整；Brightness：亮度调整；Sharpness：锐度调整
    enhancer = ImageEnhance.Color(img)

    for i in range(0, 8):
        factor = i / 2.0
        # 进行数据增强操作
        enhancer.enhance(factor).save("Color%f.png" % factor)


def t4():
    # 读取图像
    img: Image.Image = Image.open('../datas/小狗.png')
    img = img.convert("RGB")

    filters = [
        ImageFilter.MaxFilter, ImageFilter.EMBOSS, ImageFilter.BLUR, ImageFilter.CONTOUR, ImageFilter.DETAIL,
        ImageFilter.EDGE_ENHANCE, ImageFilter.EDGE_ENHANCE_MORE,
        ImageFilter.EMBOSS, ImageFilter.SMOOTH, ImageFilter.FIND_EDGES
    ]
    for fn in filters:
        img2 = img.filter(fn)
        img2.save(f"{fn.__name__}.png")

    img1 = img.convert("L").copy()
    # img1 = img
    for i in range(1):
        # BLUR: 模糊过滤器
        # CONTOUR：提取轮廓信息
        # DETAIL：提取细节的关键信息
        # EDGE_ENHANCE：边缘信息
        # EDGE_ENHANCE_MORE
        # EMBOSS： 浮雕
        # SMOOTH：光滑
        # FIND_EDGES: 边缘信息
        img1 = img1.filter(ImageFilter.EMBOSS)
    img1.show()

    # # MedianFilter: 中值滤波器，其实就是在给定区域中选择中间值作为当前区域所有像素点的值
    # # MinFilter：最小值滤波器
    # # MaxFilter: 最大值滤波器，其实就是在给定区域中选择最大值作为当前区域所有像素点的值
    #
    img = Image.open("../datas/小狗.png")
    img = img.filter(ImageFilter.MaxFilter)
    img.show()


def t5():
    # 文件图像数据转换为base64字符串
    with open('../datas/小狗.png', 'rb') as reader:
        img_data = reader.read()
    img_data = base64.b64encode(img_data)
    print(img_data)

    # base64字符串转图像
    img_data = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_data))
    img.show()

    # 图像数据转换base64字符串
    img: Image.Image = Image.open('../datas/小猫.jpg').convert("RGB")
    img_data = img.tobytes()
    img_data = base64.b64encode(img_data)
    mode = img.mode
    size = img.size
    print(img_data)

    # base64字符串转图像
    img_data = base64.b64decode(img_data)
    img = Image.frombytes(mode, size, img_data)
    img.show()

# 将图像随机保存到f'dir_path/{label}/{i}.png'中
def t6():
    to_img = transforms.ToPILImage()
    dataset = CustomLenMNIST(max_samples=20)
    random_index = np.random.permutation(len(dataset))
    for i in random_index:
        img_tensor, label = dataset[i]
        img: PIL.Image.Image = to_img(img_tensor)
        dir_path = f'../datas/MNIST/images/{label}'
        os.makedirs(dir_path, exist_ok=True)
        img.save(f'{dir_path}/{i}.png')


if __name__ == '__main__':
    t5()
