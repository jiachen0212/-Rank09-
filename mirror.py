#coding=utf-8
from PIL import Image
import os


def mirror(in_folder,out_folder):
    files = os.listdir(in_folder)    #文件夹下所有的图片的名字.
    for i in files:    #i图片的名字.
        im = Image.open(in_folder + i)
        out1 = im.transpose(Image.FLIP_LEFT_RIGHT)    #左右镜像
        out2 = im.transpose(Image.FLIP_TOP_BOTTOM)    #上下镜像
        out3 = out2.transpose(Image.FLIP_LEFT_RIGHT)  #左右上下均镜像.
        out1.save(out_folder + i[:-4] + '_1.png')
        out2.save(out_folder + i[:-4] + '_2.png')
        out3.save(out_folder + i[:-4] + '_3.png')
        im.save(out_folder + i[:-4] + '_0.png')   #原图,名字加上_0后缀.
    return
#mirror("/home/lenovo/2Tdisk/Wkyao/_/Gao_1027/1028/ex/","/home/lenovo/2Tdisk/Wkyao/_/Gao_1027/1028/mx/")