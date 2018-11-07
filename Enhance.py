#coding=utf-8
import tifffile as tiff
import scipy.misc as misc
import os
import cv2
import numpy as np
from sklearn import preprocessing
import wky_test
import scipy.io as sio
import enhance1
import mirror
import calmeanvar

FILE_2017 = '/home/lenovo/256Gdisk/juesai/2017_final.tif'
FILE_2015 = '/home/lenovo/256Gdisk/juesai/2015_final.tif'

def normal(in_file, outfile, mean, std):
    names = os.listdir(in_file)
    for name in names:
        img = tiff.imread(in_file + name)
        out = img - mean
        out = out / std
        #print out
        out = out.astype(np.float16)
        np.save(outfile + name[:-5] + '.npy', out)

FILE_15_3 = '/home/lenovo/256Gdisk/juesai/2015_3/'
e_15_3 = '/home/lenovo/256Gdisk/juesai/enhance/e15_3/'
m_15_3 = '/home/lenovo/256Gdisk/juesai/enhance/m15_3/'
FILE_17_3 = '/home/lenovo/256Gdisk/juesai/2017_3/'
e_17_3 = '/home/lenovo/256Gdisk/juesai/enhance/e17_3/'
m_17_3 = '/home/lenovo/256Gdisk/juesai/enhance/m17_3/'
FILE_15_4 = '/home/lenovo/256Gdisk/juesai/2015_4/'
e_15_4 = '/home/lenovo/256Gdisk/juesai/enhance/e15_4/'
m_15_4 = '/home/lenovo/256Gdisk/juesai/enhance/m15_4/'
FILE_17_4 = '/home/lenovo/256Gdisk/juesai/2017_4/'
e_17_4 = '/home/lenovo/256Gdisk/juesai/enhance/e17_4/'
m_17_4 = '/home/lenovo/256Gdisk/juesai/enhance/m17_4/'

nor_15_3 = '/home/lenovo/256Gdisk/juesai/enhance/normal/15_3/'
nor_15_4 = '/home/lenovo/256Gdisk/juesai/enhance/normal/15_4/'
nor_17_3 = '/home/lenovo/256Gdisk/juesai/enhance/normal/17_3/'
nor_17_4 = '/home/lenovo/256Gdisk/juesai/enhance/normal/17_4/'

#label
png_file = "/home/lenovo/256Gdisk/juesai/label_png/"
roated_file = "/home/lenovo/256Gdisk/juesai/enhance/elabel/"
mirror_file1 = "/home/lenovo/256Gdisk/juesai/enhance_4/train/annotations/training/"
mirror_file2 = "/home/lenovo/256Gdisk/juesai/enhance_3/train/annotations/training/"
enhance1.enhance(png_file, roated_file)
mirror.mirror(roated_file, mirror_file1)
mirror.mirror(roated_file, mirror_file2)

wky_test.roate(FILE_15_3, e_15_3)
wky_test.mirror(e_15_3, m_15_3)
wky_test.roate(FILE_15_4, e_15_4)
wky_test.mirror(e_15_4, m_15_4)
wky_test.roate(FILE_17_3, e_17_3)
wky_test.mirror(e_17_3, m_17_3)
wky_test.roate(FILE_17_4, e_17_4)
wky_test.mirror(e_17_4, m_17_4)

mean_15_3, var_15_3, mean_15_4, var_15_4, mean_17_3, var_17_3, mean_17_4, var_17_4 = calmeanvar.cal(FILE_2017, FILE_2015)
#normal 15 17 3 4
normal(m_15_3, nor_15_3, mean_15_3, var_15_3)
normal(m_15_4, nor_15_4, mean_15_4, var_15_4)
normal(m_17_3, nor_17_3, mean_17_3, var_17_3)
normal(m_17_4, nor_17_4, mean_17_4, var_17_4)























