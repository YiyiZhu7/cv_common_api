import torch
import cv2

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # 默认参数下加载YOLOv5s模型

# Images
img = '/Users/wanghui/PycharmProjects/yolov5/data/images/zidane.jpg'  # 更改为你的图像路径

# Inference
results = model(img)
boxes_xy = results.xyxy
boxes_xy = boxes_xy[0]

boxes = boxes_xy[:, :4]
img = cv2.imread(img)

# 对预测边框画矩形, 显示出来。
ls = []
for i in boxes:
    i_lst = i.tolist()
    ls.append(i_lst)

for (x_min, y_min, x_max, y_max) in ls:
    # 使用红色线条绘制边框，线宽为2
    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
# 显示绘制了边框的图像
cv2.imshow('Image with Boxes', img)
cv2.waitKey(0)  # 等待直到有键被按下
cv2.destroyAllWindows()

# Results
results.print()  # 打印预测信息
results.show()  # 显示图像和预测框