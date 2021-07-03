# -*- coding:utf-8 -*-
# @Time : 2021/7/2 21:28
# @Author : CY
# @File : test.py
# @Software : PyCharm

import cv2  # 导入opencv库
import numpy as np  # 导入numpy矩阵计算库


# 展示图片的方法
def showImg(name, image):
    """
    :param name: 图片输出窗口的名称
    :param image: 要展示的图片矩阵对象
    :return:
    """
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 重新定义合适的图像大小
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    :param image: 原始图像数据
    :param width: 图像指定宽度
    :param height: 图像指定高度
    :param inter:
    :return: resize后的图像矩阵

    这里是为了保证原始图片按一定比例缩放到合适大小，如果图片传入过大，
    那么在边缘检测时太多的细节就会被检测出来，干扰后面的轮廓检测
    """
    # 在调用opencv的resize函数之前，我们需要知道原始图像和后续变换后生成的图像之间的比例
    (h, w) = image.shape[:2]  # 拿到原始图像的高和宽
    if width is None and height is None:
        return image  # 未指定，直接返回
    if width is None:
        r = height / float(h)  # 通过指定的h计算图像比例
        dim = (int(w * r), height)  # 将比例传给dim
    else:
        r = width / float(w)  # 通过指定的w计算图像比例
        dim = (width, int(h * r))  # 将比例传给dim
    resized = cv2.resize(image, dim, interpolation=inter)  # 调用opencv的resize
    return resized


# 边缘检测的具体方法
def edgeDetect(image):
    """
    :param image: 经过resize处理后的图像矩阵对象
    :return:

    这一步主要是利用了opencv里的图像处理函数和著名的canny边缘检测函数
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图像转为灰度图
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯滤波函数，去除图像噪点
    edged = cv2.Canny(gray, 100, 250)  # Canny算法，检测图像边缘
    cv2.imshow("Origin Image", image)  # 显示原始图片
    cv2.imshow("Edged Image", edged)  # 显示边缘检测后的图片
    cv2.waitKey(0)  # 等待任意键退出
    cv2.destroyAllWindows()  # 销毁窗口
    return edged


# 轮廓检测的具体方法
def contourDetect(edged_image, resized_image):
    """
    :param edged_image: 进行边缘检测后得到的图片矩阵
    :param resized_image: 进行resize之后的图像矩阵
    :return:
    """
    cnts = cv2.findContours(edged_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]  # 获取到得到的外层轮廓
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]  # 根据获取到的外层轮廓计算每个轮廓的面积大小进行排序
    for c in cnts:  # 遍历轮廓值
        # 由于得到的图形轮廓并不一定都是矩形，因此对图像做一个近似处理
        peri = cv2.arcLength(c, True)  # 设置近似的
        # || approxPolyDP函数的参数解释 ||
        # 1、c：遍历传入的轮廓。
        # 2、从原始轮廓到最大轮廓的一个距离，这是一个用于控制对轮廓做近似到什么程度的参数，近似的概念不懂的可以百度一下。
        # 数值越大，则生成的近似轮廓会越规整，但准度会差，越小则近似轮廓越接近原始图像；一般先计算出边缘长度，
        # 用长度的百分之多少来决定精度的结果。
        # 3、True表示生成的近似图形是封闭的。
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 对轮廓进行近似处理
        if len(approx) == 4:  # 当轮廓近似完成后为四个点，则拿出近似轮廓
            screenCnt = approx
            break
        if len(approx) != 4:
            print("未检测到矩形，请重新传入图片")
    cv2.drawContours(resized_image, [screenCnt], -1, (0, 255, 0), 2)
    showImg("outLine", resized_image)


def new(a):
    return a
