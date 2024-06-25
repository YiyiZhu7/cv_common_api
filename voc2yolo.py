# -*- coding: utf-8 -*-
"""
定义一个voc到yolo数据格式的转换代码
voc数据标定的时候给定的是：左上角、右下角的坐标(xmin ymin xmax ymax)
yolo数据标定的时候给定的是: 中心点坐标、宽度、高度(cx cy w h)，并且是和原始图像width和height的百分比
"""

import os
from tqdm import tqdm
from xml.etree import ElementTree as ET

if __name__ == '__main__':
    whs = []
    label_2_id = {
        'plate': 0
    }
    img_dir = r"D:\datas\plate_images\images"
    voc_label_dir = r"D:\datas\plate_images\vocs"
    yolo_label_dir = r"D:\datas\plate_images\labels"
    if not os.path.exists(yolo_label_dir):
        os.makedirs(yolo_label_dir)

    # 读取所有的voc xml文件名称列表
    voc_label_names = os.listdir(voc_label_dir)
    # 遍历处理每个xml文件
    for voc_label_name in tqdm(voc_label_names):
        # 1. 解析xml文件
        tree = ET.parse(os.path.join(voc_label_dir, voc_label_name))
        # 2. 获取得到xml的根节点信息
        root = tree.getroot()
        # 3. 获取文件名称
        filename = root.find('filename').text
        # 4. 判断图像文件是否存在
        img_file = os.path.join(img_dir, filename)
        if not os.path.exists(img_file):
            continue
        # 5. 将标注的box信息转换成yolo格式，然后保存到txt文件中
        # 5.1 提取图像大小
        size_obj = root.find('size')
        width = float(size_obj.find('width').text)
        height = float(size_obj.find('height').text)

        # 5.2 遍历数据
        yolo_label_file = os.path.join(yolo_label_dir, f"{os.path.splitext(filename)[0]}.txt")
        with open(yolo_label_file, 'w', encoding='UTF-8') as writer:
            for obj in root.findall('object'):
                # a. 提取标签名称
                label = obj.find('name').text
                # b. 提取左上角、右下角坐标
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                # c. 转换成中心点坐标、宽度、高度
                w = xmax - xmin
                h = ymax - ymin
                whs.append((w, h))
                x = xmin + w / 2.0
                y = ymin + h / 2.0
                w = w / width
                h = h / height
                x = x / width
                y = y / height
                # d. 输出
                writer.writelines(f'{label_2_id[label]} {x} {y} {w} {h}\n')

    import numpy as np
    from sklearn.cluster import KMeans
    import matplotlib.pyplot   as plt

    whs = np.asarray(whs)
    model = KMeans(n_clusters=9)
    model.fit(whs)
    print(model.cluster_centers_)
    print(whs.min(axis=0))
    print(whs.max(axis=0))

    plt.scatter(whs[:, 0], whs[:, 1])
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='r')
    plt.show()
