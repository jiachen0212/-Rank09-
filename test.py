#coding=utf-8
import cv2
import tifffile as tiff
import scipy.misc as misc
import numpy as np


def Two(image):
    iTwo = image.copy()
    return iTwo


def Corrode(image):  # 腐蚀
    size = image.shape
    h = size[0]
    w = size[1]
    iCorrode = np.zeros(image.shape, np.uint8)
    kH = range(2) + range(h - 2, h)
    kW = range(2) + range(w - 2, w)
    for i in range(h):
        for j in range(w):
            if i in kH or j in kW:
                iCorrode[i, j] = 255
            elif image[i, j] == 255:
                iCorrode[i, j] = 255
            else:
                a = []
                for k in range(5):
                    for l in range(5):
                        a.append(image[i - 2 + k, j - 2 + l])
                if max(a) == 255:
                    iCorrode[i, j] = 255
                else:
                    iCorrode[i, j] = 0
    return iCorrode


def Expand(image):
    size = image.shape
    h = size[0]
    w = size[1]
    iExpand = np.zeros(image.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            iExpand[i, j] = 255
    for i in range(h):
        for j in range(w):
            if image[i, j] == 0:
                for k in range(5):
                    for l in range(5):
                        if -1 < (i - 2 + k) < h and -1 < (j - 2 + l) < w:
                            iExpand[i - 2 + k, j - 2 + l] = 0
    return iExpand


FILE_2017 = '/home/lenovo/2Tdisk/Wkyao/_/2017/2017.tif'
im_2017 = tiff.imread(FILE_2017)
print (im_2017.shape)
Img = []
for i in range(30):     #分割成30张小图进入网络进行分割.
    b = im_2017[0:5106, i * 500:i * 500 + 500]
    iTwo = Two(b)
    iCorrode = Corrode(iTwo)
    out = Expand(iCorrode)
    Img.append(out)
c = im_2017[0:5160, 15000:]
iTwo = Two(c)
iCorrode = Corrode(iTwo)
out = Expand(iCorrode)
Img.append(out)

cv2.imwrite("/home/lenovo/2Tdisk/Wkyao/_/2017/2017_1.tif", out)




                

