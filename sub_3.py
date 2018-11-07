#coding=utf-8
import tifffile as tiff
import scipy.misc as misc
import os
import cv2
import numpy as np
from sklearn import preprocessing


path2017 = '/home/lenovo/256Gdisk/juesai/enhance/normal/17_3/'
path2015 = '/home/lenovo/256Gdisk/juesai/enhance/normal/15_3/'

imgs2015 = os.listdir(path2015)
imgs2017 = os.listdir(path2017)
print (len(imgs2017), len(imgs2015))
for img2015 in imgs2015:
    for img2017 in imgs2017:
        if img2015 == img2017:
            image2015 = np.load(path2015 + img2015)#float16
            image2015 = image2015.astype(np.float)#float8
            image2017 = np.load(path2017 + img2017) #float16
            sub = image2017 - image2015
            #print sub
            np.save('/home/lenovo/256Gdisk/juesai/enhance_3/train/images/training/%s' %img2015, sub)












