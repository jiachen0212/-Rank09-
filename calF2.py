#coding=utf-8
import tifffile as tiff

def caltp(label, img):
    tp = 0
    size = label.shape
    for i in range(size[0]):
        for j in range(size[1]):
            if label[i, j] == 1:
                if img[i, j] == 1:
                    tp += 1

    return tp



