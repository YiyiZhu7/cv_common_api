# -*- coding: utf-8 -*-
import json
import os

root_dir = r"./datasets/17flowers_keypoints"
output_dir = r"./datasets/17flowers_keypoints_labels"
# output_dir = r"D:\workspaces\study\CVProject01\detection\yolov8\datasets\17flowers_keypoints\labels"
os.makedirs(output_dir, exist_ok=True)
names = os.listdir(root_dir)
label_name2idx = {}
for name in names:
    json_file = os.path.join(root_dir, name)
    with open(json_file, 'r', encoding='utf-8') as json_reader:
        json_data = json.load(json_reader)
    # print(json_data)

    # 开始提取数据
    h = json_data['imageHeight']
    w = json_data['imageWidth']
    shapes = json_data['shapes']
    if len(shapes) == 0:
        continue
    shapes = sorted(shapes, key=lambda t: t['label'])  # c1在前，c2在后 --> 为了后续代码处理的时候，在处理c2的时候，一定能够找到对应的c1
    results = []
    for obj_shape in shapes:
        obj_label = obj_shape['label']  # 取值只有两种: 框 、 关键点
        if obj_label == 'c1':  # 框
            x1, y1 = obj_shape['points'][0]
            x2, y2 = obj_shape['points'][1]
            x1 = x1 / w
            y1 = y1 / h
            x2 = x2 / w
            y2 = y2 / h
            result = [0, x1, y1, x2, y2] + [0, 0, 0] * 1
            results.append(result)
        elif obj_label == 'c2':  # 关键点
            # 找到这个关键点属于具体哪个目标框
            kx1, ky1 = obj_shape['points'][0]
            kx1 = kx1 / w
            ky1 = ky1 / h
            for c1_obj in results:
                if (c1_obj[1] <= kx1 <= c1_obj[3]) and (c1_obj[2] <= ky1 <= c1_obj[4]):
                    if c1_obj[7] > 0:
                        raise ValueError("存在重复的关键点..")
                    c1_obj[5] = kx1
                    c1_obj[6] = ky1
                    c1_obj[7] = 2
                    break
        else:
            raise ValueError(f"当前不支持该类别的标签:{obj_shape}")

    # 输出转换
    with open(os.path.join(output_dir, f"{os.path.splitext(name)[0]}.txt"), "w", encoding="utf-8") as writer:
        for result in results:
            x1, y1, x2, y2 = result[1:5]
            result[3] = x2 - x1  # width
            result[4] = y2 - y1  # height
            result[1] = x1 + result[3] / 2.0  # center x
            result[2] = y1 + result[4] / 2.0  # center y

            line = " ".join(map(str, result))
            writer.writelines(f"{line}\n")
