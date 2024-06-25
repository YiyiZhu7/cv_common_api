# -*- coding: utf-8 -*-
import binascii
import logging
import os
from flask import Flask, request, jsonify

from . import predictor

model_path = os.path.abspath(os.path.join(__file__, "..", "..", "yolov5s-plates.onnx"))
app = Flask(__name__)
p = predictor.PredictorWithOpenCV(
    model_path=model_path,
    size=640, conf=0.2, iou=0.5, max_det=1, nc=1
)


@app.route("/")
@app.route("/index")
def index():
    return "Flask App Model Predictor YOLOv5 With Plate"


@app.route("/plate/predict", methods=['POST'])
def predict():
    try:
        _img = request.form.get('image', None)  # 图像二进制数据的base64编码数据格式
        if _img is None:
            return jsonify({'code': 2, 'msg': f'请给定有效image参数，当前参数为空.'})
        r = p.predict(_img)
        if r is None or len(r) == 0:
            return jsonify({'code': 0, 'msg': '没有检查到有效物体', 'data': []})
        else:
            return jsonify({'code': 0, 'msg': f'检测到物体:{len(r)}', 'data': r})
    except binascii.Error as e:
        logging.error(f"base64字符串转换图像异常:{e}", exc_info=e)
        return jsonify({'code': 3, 'msg': f'base64字符串转换图像异常:{e}'})
    except Exception as e:
        logging.error(f"服务器异常:{e}", exc_info=e)
        return jsonify({'code': 3, 'msg': f'服务器异常:{e}'})


def run():
    app.run(host="0.0.0.0", port=9002)
