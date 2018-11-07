#coding=utf-8
import cv2
from PIL import Image
import os

def enhance(in_folder, out_folder):
    img_names = os.listdir(in_folder)    #img_names为所有图像的名字.
    for name in img_names:
        Img = Image.open(in_folder + name)
        Img.save(out_folder + name[:-4] + '_0.png')
        im = cv2.imread(in_folder + name, -1)
        im_rotate1 = im.transpose(1, 0)     #利用转置实现旋转90°(但这时候左右同时也对换了,相当于镜像了).
        cv2.imwrite(out_folder + name[:-4] + '_1.png', im_rotate1)
        img1 = Image.open(out_folder + name[:-4] + '_1.png')
        img1_1 = img1.transpose(Image.FLIP_LEFT_RIGHT)  # 再一次左右镜像,得到旋转90°后的图像.
        img1_1.save(out_folder + name[:-4] + '_1.png')
        im_rotate2 = Img.rotate(180)
        im_rotate2.save(out_folder + name[:-4] + '_2.png')
        im3 = cv2.imread(out_folder + name[:-4] + '_2.png', -1)
        im_rotate3 = im3.transpose(1,0)
        cv2.imwrite(out_folder + name[:-4] + '_3.png', im_rotate3)
        img3 = Image.open(out_folder + name[:-4] + '_3.png')
        img3 = img3.transpose(Image.FLIP_LEFT_RIGHT)  # 左右镜像
        img3.save(out_folder + name[:-4] + '_3.png')
    return
#enhance("/home/lenovo/2Tdisk/Wkyao/FirstSetp/ChenJ/1/", "/home/lenovo/2Tdisk/Wkyao/FirstSetp/ChenJ/2/")


