# Image-Scan
## 最近在学习opencv和ocr识别，因此有了这个项目，项目参考了b站https://www.bilibili.com/video/BV1oJ411D71z?p=10
这个老师讲得非常透彻，但代码不是很容易理解，因此我修改了下代码层级，将实现的主要功能均封装成对应函数了，清晰明了，并且都每一行代码都添加了注释，更加容易理解。

## 目前实现功能：
## 1、边缘检测
## 2、轮廓检测
## 3、图形变换
## 4、OCR识别
## 代码基本完成

## 如果觉得有用的话，请不要吝啬点一下关注！！谢谢
有不懂的地方，欢迎交流。QQ：837849938

# 使用方法：
说明：
1、image包下放的就是原始需要扫描的图片
2、target_image下为扫描后生成的文件
3、main包下为主函数和功能函数文件
tips：请保证图片中背景与纸张的边缘轮廓分离，清晰可见无干扰，比如使用黑色的背景拍照，提高边缘识别率

## 仅需将工程（包括image，main，target_iamge三个包）拷贝到任意ide中，目录层级不变，将需要扫描的图片拷贝到image包下，执行主函数即可，生成的扫描图片和原始图片同名，但会保存在target_image包下。其中主函数流程均有注释说明，可选择注释或开启部分功能。
