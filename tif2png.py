# coding=utf-8
from libtiff import TIFF
from scipy import misc
import numpy as np
import tifffile as tiff
import os


def tiff_to_image_array(in_folder, out_folder):
    files = os.listdir(in_folder)
    for i in files:
        print in_folder + i
        tif = tiff.imread(in_folder + i)  # 打开图像.
        tif = tif.astype(np.uint16)  # 转成16位.
        tiff.imsave(in_folder + i, tif)  # 保存.
        tif = TIFF.open(in_folder + i, mode="r")
        for im in list(tif.iter_images()):
            im_name = out_folder + i[:-5] + '.png'  # i[:-5] 截取图像的名字(不包含后缀.tiff的部分)
            misc.imsave(im_name, im)
            # cv2.imwrite(im_name, tif)
    return


tiff_to_image_array("/home/lenovo/2Tdisk/Wkyao/_/1/", "/home/lenovo/2Tdisk/Wkyao/_/2/")
