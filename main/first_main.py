# -*- coding:utf-8 -*-
# @Time : 2021/7/2 22:14
# @Author : CY
# @File : first_main.py
# @Software : PyCharm
import cv2
import scan
import ocr

"""
 整体运行逻辑流程：
    1、边缘检测
    2、轮廓检测
    3、图像变换
    4、OCR识别
"""

if __name__ == "__main__":

    image_list = scan.scan_read_allImage()  # 获取当前image包内的图片名列表
    for i in image_list:    # 循环处理已获取到的图片

        # 读入图像，记住该图像大小和缩放的比例
        image = cv2.imread("../image/" + i)  # 读入对应要扫描的数据图像文件
        if image.shape[0] <= 1000:  # 对原图长度坐标除以对应像素，得到长度比例
            ratio = image.shape[0] / 300.0
            origin = image.copy()  # 对当前原始图像进行复制，后续坐标变换会使用到
            # resize图像大小
            resized_image = scan.resize(origin, height=300)  # 通过该函数对图像进行resize操作
        elif image.shape[0] <= 3000:
            ratio = image.shape[0] / 500.0
            origin = image.copy()
            resized_image = scan.resize(origin, height=500)
        elif image.shape[0] < 5000:
            ratio = image.shape[0] / 700.0
            origin = image.copy()
            resized_image = scan.resize(origin, height=700)
        else:
            ratio = image.shape[0] / 1100.0
            origin = image.copy()
            resized_image = scan.resize(origin, height=1100)
        # 边缘检测
        edged_image = scan.edgeDetect(resized_image)  # 通过自定义函数进行边缘检测

        # 轮廓检测（只保留外圈大轮廓）
        pts = scan.contourDetect(edged_image, resized_image)

        # 透视变换
        wraped = scan.fourPointTransform(origin, pts.reshape(4, 2) * ratio)

        # 二值处理
        wraped = cv2.cvtColor(wraped, cv2.COLOR_BGR2GRAY)
        ref = cv2.threshold(wraped, 133, 245, cv2.THRESH_BINARY)[1]
        cv2.imwrite("../target_image/" + i, wraped)  # 保存该图片
        scan.showImg("Scanned", scan.resize(wraped, height=1000))
        # 进行ocr识别
        # ocr.ocr_judge(i)

    # 读取处理好的图片，并连接生成pdf文档
    # scan.save_image_by_pdf()





