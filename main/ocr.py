# -*- coding:utf-8 -*-
# @Time : 2021/7/4 20:46
# @Author : CY
# @File : ocr.py
# @Software : PyCharm

# 1、首先下载tesseract的安装包,配置环境变量，该包用于ocr识别
# 2、使用代码操作这个ocr包还需要安装一个python工具集，代码如下：
# pip install pytesseract

# 导入必要的包
from PIL import Image
import pytesseract
import cv2
import os
import scan


"""
对主函数中经过变换后处理好的图片进行读写然后进行ocr识别
注意：目前我的ocr并未经过字库训练，为默认的字库（仅支持全英文识别）
     因此pull下代码后执行很可能会得不到自己想要的结果。
     如果要手写识别的话，后面需要通过另外的工具对tesseract进行训练，
     识别效果才会大大提升！
"""


# ocr识别判断
def ocr_judge():
    preprocess = 'blur'

    image = cv2.imread("../target_image/scanned.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if preprocess == 'thresh':
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if preprocess == 'blur':
        gray = cv2.medianBlur(gray, 3)
        filename = "{}.png".format(os.getpid())
    cv2.imwrite('../target_image/' + filename, gray)

    text = pytesseract.image_to_string(Image.open('../target_image/' + filename))
    print(text)
    os.remove('../target_image/' + filename)

    cv2.imshow("Image", scan.resize(image, height=1000))
    cv2.imshow("output", scan.resize(gray, height=1000))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
