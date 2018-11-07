#coding=utf-8
import tifffile as tiff
import scipy.misc as misc
import os
import cv2
import numpy as np
from sklearn import preprocessing
file15 = '/home/lenovo/256Gdisk/juesai/2015_final.tif'
file17 = '/home/lenovo/256Gdisk/juesai/2017_final.tif'
img15 = tiff.imread(file15).transpose([1, 2, 0])
img15 = img15[:,:,3]
img17 = tiff.imread(file17).transpose([1, 2, 0])
img17 = img17[:,:,3]

New_2015 = preprocessing.scale(img15)
New_2015 = New_2015.astype(np.float)
New_2017 = preprocessing.scale(img17)
New_2017 = New_2017.astype(np.float16)

sub = New_2017 - New_2015
size = sub.shape
np.save('/home/lenovo/256Gdisk/tainchi/juesaidata/test.npy', sub)












