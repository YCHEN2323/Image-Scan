# -*- coding:utf-8 -*-
# @Time : 2021/7/2 21:28
# @Author : CY
# @File : scan.py
# @Software : PyCharm

import cv2  # 导入opencv库
import numpy as np  # 导入numpy矩阵计算库
import os


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
    edged = cv2.Canny(gray, 50, 250)  # Canny算法，检测图像边缘
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
    screenCnt = None
    for c in cnts:  # 遍历轮廓值
        # 由于得到的图形轮廓并不一定都是矩形，因此对图像做一个近似处理
        peri = cv2.arcLength(c, True)  # 设置近似的
        # || approxPolyDP函数的参数解释 ||
        # 1、c：遍历传入的轮廓。
        # 2、从原始轮廓到最大轮廓的一个距离，这是一个用于控制对轮廓做近似到什么程度的参数，近似的概念不懂的可以百度一下。
        # 数值越大，则生成的近似轮廓会越规整，但准度会差，越小则近似轮廓越接近原始图像；一般先计算出边缘长度，
        # 用长度的百分之多少来决定精度的结果。
        # 3、True表示生成的近似图形是封闭的。
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)  # 对轮廓进行近似处理
        if len(approx) == 4:  # 当轮廓近似完成后为四个点，则拿出近似轮廓
            screenCnt = approx
            break
        if len(approx) != 4:
            print("contourDetect未检测到矩形")
    cv2.drawContours(resized_image, [screenCnt], -1, (0, 255, 0), 2)
    showImg("outLine", resized_image)
    return screenCnt


# 四点计算
def order_point(pts):
    """
    :param pts:  四个坐标值和ratio比例值的乘积(输入坐标)
    :return: 返回计算好的坐标值
    """
    # 四个坐标点
    rect = np.zeros((4, 2), dtype="float32")
    # 按顺序找到对应四个角的坐标点，分别为左上、右上、左下、右下。
    s = pts.sum(axis=1)
    # 计算左上、右下
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算右上、左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 透视变换处理
def fourPointTransform(origin, pts):
    """
    :param origin: 输入的原始图像
    :param pts: 四个坐标值和ratio比例值的乘积(输入坐标)
    :return:    返回计算后的图像
    """
    recen = order_point(pts)  # 进行四点坐标的获取
    (tL, tR, bR, bL) = recen  # 分别读取四点坐标

    # 根据四点坐标计算输入的宽和高值
    widthA = np.sqrt(((bR[0] - bL[0]) ** 2) + ((bR[1] - bL[1]) ** 2))
    widthB = np.sqrt(((tR[0] - tL[0]) ** 2) + ((tR[1] - tL[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))  # 确保误差最小，使用长的边，因此找出计算后的最大值
    heightA = np.sqrt(((tR[0] - bR[0]) ** 2) + ((tR[1] - bR[1]) ** 2))
    heightB = np.sqrt(((tL[0] - bL[0]) ** 2) + ((tL[1] - bL[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置
    dts = np.array([  # dts表示变换完成后的输出坐标
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(recen, dts)
    warped = cv2.warpPerspective(origin, M, (maxWidth, maxHeight))
    return warped  # 返回计算结果


# 读取当前‘image‘包下的图片
def scan_read_allImage():
    """
    :return: 返回一个包含image包下所有文件名的数组
    """
    lists = os.listdir('../image')
    if len(lists) > 0:
        print("已获取到image包下的图片：")
        count = 1
        for i in lists:
            print(str(count) + "、" + i)
            count += 1
        return lists
    else:
        print("读取image包失败")


# 将生成的图片连接成长串并生成pdf
def save_image_by_pdf():
    lists = os.listdir('../target_image')
    result_image_list = []
    for i in lists:
        image = cv2.imread("../target_image/" + i)
        showImg("result", image)
        result_image_list.append(image)
    result = np.hstack(result_image_list)
    showImg("result", result)
