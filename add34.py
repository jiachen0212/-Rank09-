#coding=utf-8
import numpy as np
import tifffile as tiff
import crf_c as crf
import cv2
import calF1
import calF2


# FILE_new_2017 = '/home/lenovo/2Tdisk/Wkyao/_/20171105_quarterfinals/quarterfinals_2017.tif'
# new_2017 = tiff.imread(FILE_new_2017).transpose([1, 2, 0])
# img17_1 = new_2017[:, :, 2]
# size = img17_1.shape
# img17_2 = new_2017[:, :, 3]
# img17 = (img17_1 + img17_2) / 2
#
# vgg_3_npy = '/home/lenovo/256Gdisk/tainchi/vgg/1120_3.npy'
# vgg_4_npy = '/home/lenovo/256Gdisk/tainchi/vgg/1120_4.npy'
# vggnpy3 = np.load(vgg_3_npy)
# vggnpy4 = np.load(vgg_4_npy)
#
def open_and_close(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    img = cv2.dilate(img, kernel)
    return  img
#
# img1 = np.argmax(vggnpy3, 0).astype(np.uint8)
# img2 = np.argmax(vggnpy4, 0).astype(np.uint8)
# size = img1.shape
# for i in range(size[0]):
#     for j in range(size[1]):
#         img1[i, j] = img1[i, j] or img2[i, j]
# tiff.imsave('/home/lenovo/256Gdisk/tainchi/merge/add1120.tif', img1)









file1 = tiff.imread('/home/lenovo/256Gdisk/tainchi/vgg/fo_crf_1120_4.tif')
file2 = tiff.imread('/home/lenovo/256Gdisk/tainchi/vgg/fo_crf_1120_3.tif')
size = file1.shape
for i in range(size[0]):
    for j in range(size[1]):
        file1[i, j] = file1[i, j] or file2[i, j]     #or 或运算
tiff.imsave('/home/lenovo/256Gdisk/tainchi/merge/fo_crfadd1120.tif', file1)
# Img = tiff.imread('/home/lenovo/256Gdisk/tainchi/merge/crfadd1120.tif')
openimg = open_and_close(file1)
tiff.imsave('/home/lenovo/256Gdisk/tainchi/merge/fo_crfaddopen1120last.tif', openimg)











































































