# -*- coding: utf-8 -*-
import base64

import requests


def image_base64(image_path, print_data=True):
    with open(image_path, "rb") as reader:
        img_content = reader.read()  # 加载图像的所有二进制数据
        img_base64_content = base64.b64encode(img_content)  # 编码
        if print_data:
            print(img_base64_content)
        return img_base64_content


def invoke_remote_api(image_path):
    response = requests.post(
        url='http://121.40.96.93:9998/face/search',
        data={
            'image': image_base64(image_path, print_data=False),
            'k': 3,
            'threshold': 0.8
        }
    )
    if response.status_code == 200:
        result = response.json()
        print(result)
        print(type(result))
    else:
        raise ValueError("调用服务器异常")


if __name__ == '__main__':
    # image_base64(
    #     "./datas/features/images/Adolfo_Rodriguez_Saa/Adolfo_Rodriguez_Saa_0002.jpg"
    # )
    # image_base64(
    #     "./datas/features/images/zxc/img1.jpeg"
    # )
    invoke_remote_api(
        "./datas/features/images/zxc/img1.jpeg"
    )
