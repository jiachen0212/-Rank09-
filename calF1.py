#coding=utf-8
import tifffile as tiff

def calf1(label, img):
    beta = 2
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    size = label.shape
    for i in range(size[0]):
        for j in range(size[1]):
            if label[i, j] == 1:
                if img[i, j] == 1:
                    tp += 1
                else:
                    fn += 1
            if label[i, j] == 0:
                if img[i, j] == 1:
                    fp += 1
                else:
                    tn += 1
    print (tp, tn, fp, fn)
    P = float(tp) / (tp + fp)
    R = float(tp) / (tp + fn)
    F1 = 2 * (P * R) * (1 + beta * beta) / (beta * beta * P + R)
    return F1



