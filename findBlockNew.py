# encoding:utf-8
from libtiff import TIFF
from scipy import misc
import numpy as np
import tifffile as tiff
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

# 存放tiff切割图的路径
path_2017_4 = '/home/lenovo/256Gdisk/juesai/2017_4/'
path_2015_4 = '/home/lenovo/256Gdisk/juesai/2015_4/'
path_2017_3 = '/home/lenovo/256Gdisk/juesai/2017_3/'
path_2015_3 = '/home/lenovo/256Gdisk/juesai/2015_3/'
path_label = '/home/lenovo/256Gdisk/juesai/label/'

# 决赛数据
new_FILE_2017 = '/home/lenovo/256Gdisk/juesai/2017_final.tif'
new_FILE_2015 = '/home/lenovo/256Gdisk/juesai/2015_final.tif'



FILE_netout = '/home/lenovo/256Gdisk/tainchi/merge/fo_crfaddopen1120.tif'      # 网络跑图!!!!!

im_2017 = tiff.imread(new_FILE_2017).transpose([1, 2, 0])
im_2015 = tiff.imread(new_FILE_2015).transpose([1, 2, 0])
# im_biaozhu = tiff.imread(FILE_biaozhu).astype(np.uint8)
size = im_2017.shape

# biaozhu = cv2.imread(FILE_biaozhu)  # 一定要用cv2来读tif


# netout = cv2.imread(FILE_netout)
netout = tiff.imread(FILE_netout)
netout = netout.astype(np.bool)
netout = netout.astype(np.uint8)
# biaozhu = np.array([np.mean(im_biaozhu, axis=2)]).transpose(1, 2, 0)


def findBlockNew():     # 网络跑的图
    img, contours, h = cv2.findContours(netout, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print contours
    print len(contours), '******************'  # 之前contours是list,现在变成了tuple,所以要取contours[1]才是之前的contours
    # print contours[0]
    # print contours[1]
    num = 0
    for i in range(len(contours)):
        b = cv2.boundingRect(contours[i])
        if b[2] * b[3] > 5000:  # 大于一定面积才截图
            # print b  # 外框矩形信息
            ax = (b[0] + b[2] / 2, b[1] + b[3] / 2)
            if not (ax[0] < 128 or ax[1] < 128 or ax[0] > size[1] - 128 or ax[1] > size[0] - 128):  # 边上的无法截取
                num = num + 1
                box_2017_4 = im_2017[ax[1] - 128:ax[1] + 128, ax[0] - 128:ax[0] + 128, 3]
                box_2015_4 = im_2015[ax[1] - 128:ax[1] + 128, ax[0] - 128:ax[0] + 128, 3]
                box_2017_3 = im_2017[ax[1] - 128:ax[1] + 128, ax[0] - 128:ax[0] + 128, 2]
                box_2015_3 = im_2015[ax[1] - 128:ax[1] + 128, ax[0] - 128:ax[0] + 128, 2]
                box_biaozhu = netout[ax[1] - 128:ax[1] + 128, ax[0] - 128:ax[0] + 128]

                box_2017 = im_2017[ax[1] - 128:ax[1] + 128, ax[0] - 128:ax[0] + 128, :3]
                box_2015 = im_2015[ax[1] - 128:ax[1] + 128, ax[0] - 128:ax[0] + 128, :3]

                show(box_biaozhu, box_2017, box_2015, num)
                tiff.imsave(path_2017_4 + 'x_%d.tiff' % num, box_2017_4)
                tiff.imsave(path_2015_4 + 'x_%d.tiff' % num, box_2015_4)
                tiff.imsave(path_2017_3 + 'x_%d.tiff' % num, box_2017_3)
                tiff.imsave(path_2015_3 + 'x_%d.tiff' % num, box_2015_3)
                tiff.imsave(path_label + 'x_%d.tiff' % num, box_biaozhu)

                # print ax  # 坐标
                print num, 'nums'  # 截图个数

                # print num


def findBiaozhu():      # 人工标注图
    FILE_biaozhu = '/home/lenovo/256Gdisk/juesai/biaozhu_1111.tif'  # 人工标注!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
    biaozhu = tiff.imread(FILE_biaozhu)
    # biaozhu = np.mean(biaozhu, axis=2)  # 这里需要mean掉第三通道
    biaozhu = biaozhu.astype(np.bool)
    biaozhu = biaozhu.astype(np.uint8)

    img, contours, h = cv2.findContours(biaozhu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print contours
    print len(contours), '******************'  # 之前contours是list,现在变成了tuple,所以要取contours[1]才是之前的contours
    # print contours[0]
    # print contours[1]
    num = 0
    for i in range(len(contours)):
        b = cv2.boundingRect(contours[i])
        if b[2] * b[3] > 100:  # 大于一定面积才截图
            # print b  # 外框矩形信息
            ax = (b[0] + b[2] / 2, b[1] + b[3] / 2)
            if not (ax[0] < 128 or ax[1] < 128 or ax[0] > size[1] - 128 or ax[1] > size[0] - 128):  # 边上的无法截取
                num = num + 1
                box_2017_4 = im_2017[ax[1] - 128:ax[1] + 128, ax[0] - 128:ax[0] + 128, 3]
                box_2015_4 = im_2015[ax[1] - 128:ax[1] + 128, ax[0] - 128:ax[0] + 128, 3]
                box_2017_3 = im_2017[ax[1] - 128:ax[1] + 128, ax[0] - 128:ax[0] + 128, 2]
                box_2015_3 = im_2015[ax[1] - 128:ax[1] + 128, ax[0] - 128:ax[0] + 128, 2]
                box_biaozhu = biaozhu[ax[1] - 128:ax[1] + 128, ax[0] - 128:ax[0] + 128]

                tiff.imsave(path_2017_4 + 'ge_%d.tiff' % num, box_2017_4)
                tiff.imsave(path_2015_4 + 'ge_%d.tiff' % num, box_2015_4)
                tiff.imsave(path_2017_3 + 'ge_%d.tiff' % num, box_2017_3)
                tiff.imsave(path_2015_3 + 'ge_%d.tiff' % num, box_2015_3)
                tiff.imsave(path_label + 'ge_%d.tiff' % num, box_biaozhu)

                # cv2.imwrite('/home/wkyao_check/_/2017_4/x_%d.tiff' % num, box_2017_4)
                # cv2.imwrite('/home/wkyao_check/_/2015_4/x_%d.tiff' % num, box_2015_4)
                # cv2.imwrite('/home/wkyao_check/_/label_tiny/x_%d.tiff' % num, box_biaozhu)

                # print ax  # 坐标
                print num, '*******'  # 截图个数

                        # print num


def tiff_to_image_array(in_folder, out_folder):
    files = os.listdir(in_folder)
    for i in files:
        # print in_folder + i
        tif = tiff.imread(in_folder + i)  # 打开图像.
        tif = tif.astype(np.uint16)  # 转成16位.
        tiff.imsave(in_folder + i, tif)  # 保存.
        tif = TIFF.open(in_folder + i, mode="r")
        # tif = tif.astype(np.float16)
        for im in list(tif.iter_images()):
            im_name = out_folder + i[:-5] + '.png'  # i[:-5] 截取图像的名字(不包含后缀.tiff的部分)
            misc.imsave(im_name, im)


def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix


def show(box_biaozhu, box_2017, box_2015, name):
    plt.subplots(ncols=3, nrows=1, figsize=(12, 6))

    # 创建一个mask
    mt = np.ma.array(np.ones((256, 256), dtype=np.uint8),  # 500 x 500的全1矩阵
                     mask=(box_biaozhu / np.max(box_biaozhu)))

    p1 = plt.subplot(231)
    i1 = p1.imshow(scale_percentile(box_2015))

    p2 = plt.subplot(232)
    i2 = p2.imshow(scale_percentile(box_2017))
    p2.imshow(mt, alpha=0.3, vmin=0, vmax=1)

    p3 = plt.subplot(233)
    i3 = p3.imshow(box_biaozhu)
    plt.title(name)
    plt.show()


if __name__ == '__main__':
    # findBiaozhu()  # 人工标注先不要!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    findBlockNew()   # 网络跑图
    # tiff_to_image_array(path_2015_4, '/home/lenovo/256Gdisk/tainchi/1/2015_3_png/')
    # tiff_to_image_array(path_2017_4, '/home/lenovo/256Gdisk/tainchi/1/2017_3_png/')
    # tiff_to_image_array(path_2015_4, '/home/lenovo/256Gdisk/tainchi/1/2015_4_png/')
    # tiff_to_image_array(path_2017_4, '/home/lenovo/256Gdisk/tainchi/1/2017_4_png/')
    tiff_to_image_array(path_label, '/home/lenovo/256Gdisk/juesai/label_png/')
