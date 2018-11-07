# encoding: utf-8
import sys
import numpy as np
import tifffile as tiff
import pydensecrf.densecrf as dcrf
import tensorflow as tf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from numpy import random

# processed_probabilities = 前向传播达到的概率值

def crf(image, softmax):
    # softmax = final_probabilities.squeeze()    squeeze() 去除size为1的维度.

    #print ('*****')
    #print (softmax.shape)    #(2, 5106, 300)
    # 输入数据应为概率值的负对数
    # 你可以在softmax_to_unary函数的定义中找到更多信息
    unary = softmax_to_unary(softmax)  # 转为一元.

    # 输入数据应为C-连续的——我们使用了Cython封装器
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(image.shape[0], image.shape[1], 2)
    #print image.shape[0], image.shape[1], image.shape[2]     #5106 300 3
    #print unary.shape      #(2, 1531800)
    d.setUnaryEnergy(unary)

    # 潜在地对空间上相邻的小块分割区域进行惩罚——促使产生更多空间连续的分割区域
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[:2])    # (5,5)  #(10,10)
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # 这将创建与颜色相关的图像特征——因为我们从卷积神经网络中得到的分割结果非常粗糙，
    # 我们可以使用局部的颜色特征来改善分割结果
    feats = create_pairwise_bilateral(sdims=(10, 10), schan=(8, 8, 8), img=image, chdim=2)   # sdims=(50, 50), schan=(20, 20, 20)坐标和RGB
    #往小调.
    d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    # d是一个惩罚项好像,是随机分类的指标.


    # print d.shape
    q = d.inference(1)    #2
    res = np.argmax(q, axis=0).reshape((image.shape[0], image.shape[1]))

    res = res.astype(np.uint8)

    return res


