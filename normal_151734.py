#coding=utf-8
import tifffile as tiff
import scipy.misc as misc
import os
import cv2
import numpy as np

img_15_3 = '/home/lenovo/256Gdisk/juesai/enhance/m15_3/'
img_15_4 = '/home/lenovo/256Gdisk/juesai/enhance/m15_4/'
img_17_3 = '/home/lenovo/256Gdisk/juesai/enhance/m17_3/'
img_17_4 = '/home/lenovo/256Gdisk/juesai/enhance/m17_4/'

nor_15_3 = '/home/lenovo/256Gdisk/juesai/enhance/normal/15_3/'
nor_15_4 = '/home/lenovo/256Gdisk/juesai/enhance/normal/15_4/'
nor_17_3 = '/home/lenovo/256Gdisk/juesai/enhance/normal/17_3/'
nor_17_4 = '/home/lenovo/256Gdisk/juesai/enhance/normal/17_4/'



def normal(in_file, outfile, mean, std):
    names = os.listdir(in_file)
    for name in names:
        img = tiff.imread(in_file + name)
        out = img - mean
        out = out / std
        #print out
        out = out.astype(np.float16)
        np.save(outfile + name[:-5] + '.npy', out)


normal(img_15_3, nor_15_3, 339.35, 118.36)
normal(img_15_4, nor_15_4, 490.54, 162.03)
normal(img_17_3, nor_17_3, 301.00, 147.72)
normal(img_17_4, nor_17_4, 496.02, 217.71)




