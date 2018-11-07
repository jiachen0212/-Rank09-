#coding=utf-8
import tifffile as tiff
import os
import numpy as np

def cal(FILE_2017, FILE_2015):
    img_2017 = tiff.imread(FILE_2017).transpose([1, 2, 0])
    img_2015 = tiff.imread(FILE_2015).transpose([1, 2, 0])
    img_15_3 = img_2015[:, :, 2]
    img_15_4 = img_2015[:, :, 3]
    img_17_3 = img_2017[:, :, 2]
    img_17_4 = img_2017[:, :, 3]
    mean_15_3 = np.mean(img_15_3)
    var_15_3 = np.std(img_15_3)
    mean_17_3 = np.mean(img_17_3)
    var_17_3 = np.std(img_17_3)
    mean_15_4 = np.mean(img_15_4)
    var_15_4 = np.std(img_15_4)
    mean_17_4 = np.mean(img_17_4)
    var_17_4 = np.std(img_17_4)
    print mean_15_3, var_15_3, mean_15_4, var_15_4, mean_17_3, var_17_3, mean_17_4, var_17_4
    return mean_15_3, var_15_3, mean_15_4, var_15_4, mean_17_3, var_17_3, mean_17_4, var_17_4






