#coding=utf-8
from libtiff import TIFF
from scipy import misc
import numpy as np
import tifffile as tiff
import os
import cv2

def tiff_to_image_array(in_folder, out_folder):
    files = os.listdir(in_folder)
    for i in files:
        #print in_folder + i
        tif = tiff.imread(in_folder + i)    #打开图像.
        tif = tif.astype(np.uint16)         #转成16位.
        tiff.imsave(in_folder + i, tif)     #保存.
        tif = TIFF.open(in_folder + i, mode = "r")
        #tif = tif.astype(np.float16)
        for im in list(tif.iter_images()):
            im_name = out_folder + i[:-4] + 'png'     #i[:-5] 截取图像的名字(不包含后缀.tiff的部分)
            misc.imsave(im_name, im)
            # img = cv2.imread(im_name, -1)
            # print img.shape
    return
tiff_to_image_array("/home/lenovo/2Tdisk/Wkyao/_/1/label_tiny/","/home/lenovo/2Tdisk/Wkyao/_/1/1/")
