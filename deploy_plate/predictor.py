# -*- coding: utf-8 -*-
import base64
from io import BytesIO
from typing import List, Dict, Any

import cv2 as cv
import numpy as np
import torch
from PIL import Image


# noinspection DuplicatedCode
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    ratio_pad = ((r, r), (dw, dh))

    if shape[::-1] != new_unpad:  # resize
        im = cv.resize(im, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border
    return im, ratio_pad


# noinspection DuplicatedCode
def xywh2xyxy(x):
    x[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x = center_x - w/2
    x[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y = center_y - h/2
    x[:, 2] = x[:, 0] + x[:, 2]  # bottom right x = top left x + w
    x[:, 3] = x[:, 1] + x[:, 3]  # bottom right y = top left y + h
    return x


@torch.no_grad()
def non_max_suppression(
        prediction,  # [N,M,4+1+nc] N表示批次大小，M表示预测的边框数目，nc表示类别数目；4+1+nc->先4个预测边框对应的中心点&高度宽度、1个是否包含物体的概率、nc个包含具体各个类别的对应概率
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=30,
        max_nms=300
):
    x = prediction[0]  # [N,M,4+1+nc] -> [M,4+1+nc] 先4个预测边框对应的中心点&高度宽度、1个是否包含物体的概率、nc个包含具体各个类别的对应概率
    xc = x[..., 4] > conf_thres  # candidates 获取包含物体的概率超过阈值的预测边框index/bool
    x = x[xc]  # confidence 获取包含物体的概率超过阈值的预测边框
    if not x.shape[0]:
        return None

    # Compute conf 计算预测为某类物体的概率值
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf 计算边框属于各个类别的对应概率

    # Box
    box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)

    # Detections matrix nx6 (xyxy, conf, cls) 边框坐标 + 预测类别的对应概率 + 预测类别id
    j = x[:, 5:].argmax(1).reshape(-1, 1)  # 获取每行的最大值对应的下标 类别下标
    conf = x[:, 5:].max(1, keepdims=True)
    x = np.hstack((box, conf, j))[conf.reshape(-1) > conf_thres]

    # Check shape
    n = x.shape[0]  # number of boxes
    if not n:  # no boxes
        return None
    else:
        x = x[x[:, 4].argsort()[::-1][:1]]  # sort by confidence
        return x


def scale_boxes(boxes, img0_shape, ratio_pad):
    boxes[:, [0, 2]] -= ratio_pad[1][0]  # x padding x1 x2
    boxes[:, [1, 3]] -= ratio_pad[1][1]  # y padding y1 y2
    boxes[:, :4] /= ratio_pad[0][0]

    # 边框截断
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img0_shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img0_shape[0])  # y1, y2
    return boxes


class PredictorWithOpenCV(object):
    def __init__(self, model_path, size=640, conf=0.5, iou=0.5, max_det=4, nc=80):
        """
        预测器的初始化/恢复 --> 为了保证仅初始化一次
        """
        super(PredictorWithOpenCV, self).__init__()
        self.net = cv.dnn.readNetFromONNX(model_path)  # 模型恢复加载 1,3,640,640
        self.size = size
        self.conf = conf
        self.iou = iou
        self.nc = nc
        self.max_det = 1  # 一张图像最多允许max_det个检测边框的输出
        self.max_nms = 1
        self.class_id_2_names = ['plate']  # 一般情况下应该是随着模型一起恢复的
        print(f"OpenCV模型恢复成功:{model_path} _ {self.size}")

    def predict(self, base64_image_str: str) -> List[Dict[str, Any]]:
        """
        模型预测
        """
        # 1. 将image字符串转换为image图像
        img_content = base64.b64decode(base64_image_str)  # 解码还原、
        image = Image.open(BytesIO(img_content))
        image = image.convert("RGB")  # RGB格式
        image = np.asarray(image)  # [H,W,C] C==3

        # 图像的预处理
        s = image.shape[:2]  # HWC
        image, ratio_pad = letterbox(image, [self.size, self.size])  # [H,W,C]
        im = np.ascontiguousarray(np.array([image]).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
        im = im / 255.0  # 0-255 --> 0-1.0

        # 模型预测 OpenCV模型结果返回，Python中是numpy的数组结构
        self.net.setInput(im)
        y = self.net.forward()

        # 后处理逻辑
        y = non_max_suppression(y, self.conf, self.iou, max_det=self.max_det, max_nms=self.max_nms)  # NMS
        if y is None:
            return []
        scale_boxes(y[:, :4], s, ratio_pad=ratio_pad)  # 坐标还原

        # 结果返回
        result = []
        img_h, img_w = s
        for box in y:
            result.append({
                'x1': box[0] / img_w,
                'y1': box[1] / img_h,
                'x2': box[2] / img_w,
                'y2': box[3] / img_h,
                'prob': float(box[4]),
                'class_id': int(box[5]),
                'class_name': self.class_id_2_names[int(box[5])]
            })
        return result
