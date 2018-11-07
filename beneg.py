#coding=utf-8
import cv2
import os
from numpy import *
import numpy as np

file = open("/home/lenovo/256Gdisk/juesai/neg.txt")
names = file.read()
name = names.split()   #用split()函数生成list
imgpath = '/home/lenovo/256Gdisk/juesai/label_png/'
for line in name:
	#print line
	img = cv2.imread(imgpath + 'x_' + line + '.png')
	size = img.shape
	img = zeros(size, np.uint8)
	img = np.mean(img, axis = 2)
	cv2.imwrite(imgpath + 'x_' + line + '.png', img)



