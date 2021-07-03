# -*- coding:utf-8 -*-
# @Time : 2021/7/2 22:14
# @Author : CY
# @File : first_main.py
# @Software : PyCharm
import cv2
import test

"""
 整体运行逻辑流程：
    1、边缘检测
    2、轮廓检测
    3、图像变换
    4、OCR识别
"""

if __name__ == "__main__":
    # 读入图像，记住该图像大小和缩放的比例
    image = cv2.imread("../image/paper.jpg")  # 读入对应要扫描的数据图像文件
    if image.shape[0] <= 1000:  # 对原图长度坐标除以对应像素，得到长度比例
        ratio = image.shape[0] / 200.0
        origin = image.copy()  # 对当前原始图像进行复制，后续坐标变换会使用到
    # resize图像大小
        resized_image = test.resize(origin, height=200)  # 通过该函数对图像进行resize操作
    elif image.shape[0] <= 3000:
        ratio = image.shape[0] / 500.0
        origin = image.copy()
        resized_image = test.resize(origin, height=500)
    elif image.shape[0] > 5000:
        ratio = image.shape[0] / 800.0
        origin = image.copy()
        resized_image = test.resize(origin, height=800)

    # 边缘检测
    edged_image = test.edgeDetect(resized_image)  # 通过自定义函数进行边缘检测

    # 轮廓检测（只保留外圈大轮廓）
    test.contourDetect(edged_image, resized_image)
