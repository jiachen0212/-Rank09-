# coding=utf-8
import cv2
from PIL import Image
import os
import numpy as np
import tifffile as tiff
from sklearn import preprocessing


def roate(in_folder, out_folder):
    img_names = os.listdir(in_folder)  # img_names为tiff图像
    for name in img_names:
        image = tiff.imread(in_folder + name)
        image_90 = np.rot90(image)      # 转90°
        image_180 = np.rot90(image, 2)  # 转180°
        image_270 = np.rot90(image, 3)  # 转270°
        tiff.imsave(out_folder + name[:-5] + '_1.tiff', image_90)
        tiff.imsave(out_folder + name[:-5] + '_2.tiff', image_180)
        tiff.imsave(out_folder + name[:-5] + '_3.tiff', image_270)
        tiff.imsave(out_folder + name[:-5] + '_0.tiff', image)

    return


def mirror(in_folder, out_folder):
    img_names = os.listdir(in_folder)  # img_names为tiff图像
    for name in img_names:
        image = tiff.imread(in_folder + name)
        rotate_1 = np.rot90(image.T, 3)      #左右翻转
        rotate_2 = np.rot90(image.T)         #上下翻转
        rotate_3 = np.rot90((np.rot90(image.T)).T, 3)  #上下左右翻转
        tiff.imsave(out_folder + name[:-5] + '_1.tiff', rotate_1)
        tiff.imsave(out_folder + name[:-5] + '_2.tiff', rotate_2)
        tiff.imsave(out_folder + name[:-5] + '_3.tiff', rotate_3)
        tiff.imsave(out_folder + name[:-5] + '_0.tiff', image)

    return





#label

# def roate(in_folder, out_folder):
#     img_names = os.listdir(in_folder)  # img_names为tiff图像
#     for name in img_names:
#         image = cv2.imread(in_folder + name)
#         image_90 = np.rot90(image, 1)  # 顺时针转90°
#         image_180 = np.rot90(image, 2)  # 转180°
#         image_270 = np.rot90(image, 3)  # 转270°
#         tiff.imsave(out_folder + name[:-4] + '_1.png', image_90)
#         tiff.imsave(out_folder + name[:-4] + '_2.png', image_180)  # label
#         tiff.imsave(out_folder + name[:-4] + '_3.png', image_270)
#         tiff.imsave(out_folder + name[:-4] + '_0.png', image)
#
#     return
#

# def test(in_folder1, in_folder2):
#     img_names1 = os.listdir(in_folder1)
#     img_names2 = os.listdir(in_folder2)
#     for name1 in img_names1:
#         for name2 in img_names2:
#             img1 = np.load(in_folder1 + name1)
#             #img2 = np.load(in_folder2 + name2)
#             #img = tiff.imread(in_folder + name)
#             img2 = cv2.imread(in_folder2 + name2)
#             for i in range(img1.shape[0]):
#                 for j in range(img1.shape[1]):
#                         if img1[i][j]:
#                             print (img1[i][j] - img2[i][j])
#     print '*****************************************'

#test('/home/lenovo/2Tdisk/Wkyao/FirstSetp/wkytest_scale/1/','/home/lenovo/2Tdisk/Wkyao/FirstSetp/wkytest_scale/tiff/')


path = '/home/lenovo/2Tdisk/Wkyao/FirstSetp/wkytest_scale/1/'
imgs = os.listdir(path)
# for img in imgs:
#     print img
#     im = np.load(path + img)
#     #im = im.astype(np.float16)
#     #im = tiff.imread(path + img)
#     size = im.shape
#     for i in range(size[0]):
#         for j in range(size[1]):
#             if im[i, j] > 0:
#                 print im[i, j]
#             print im[i, j]
    #tiff.imsave(path + img[:-4] + '.tiff', im)  #只能保存tiff先
    #image = cv2.imread(path + img[:-4] + '.tiff', -1)
    #cv2.imwrite(path + img[:-4] + '.jpg', image)





# path2015 = '/home/lenovo/2Tdisk/Wkyao/FirstSetp/wkytest_scale/1/'
# path2017 = '/home/lenovo/2Tdisk/Wkyao/FirstSetp/wkytest_scale/2/'
# imgs2015 = os.listdir(path2015)
# imgs2017 = os.listdir(path2017)
# for img2015 in imgs2015:
#     for img2017 in imgs2017:
#             #print img2015, img2017
#         #image2015 = tiff.imread(path2015 + img2015)
#         image2015 = np.load(path2015 + img2015)
#         #print image2015
#         #image2015 = image2015.astype(np.float32)
#         #print image2015
#         #image2017 = tiff.imread(path2017 + img2017)
#         image2017 = np.load(path2017 + img2017)
#         image2017 = image2017.astype(np.float32)   #这句是关键
#         New_2015 = preprocessing.scale(image2015)
#         New_2017 = preprocessing.scale(image2017)
#         sub = (New_2017 - New_2015)
#         #print sub
#         np.save('/home/lenovo/2Tdisk/Wkyao/FirstSetp/wkytest_scale/sub.npy', sub)

        # for i in range(sub.shape[0]):
        #     for j in range(sub.shape[1]):
        #         if sub[i, j]:
        #             print sub




























