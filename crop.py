#coding=utf-8
import numpy as np
import cv2
import tifffile as tiff

fusai_FILE_2017 = '/home/lenovo/2Tdisk/Wkyao/_/20171105_quarterfinals/quarterfinals_2017.tif'
fusai_FILE_2015 = '/home/lenovo/2Tdisk/Wkyao/_/20171105_quarterfinals/quarterfinals_2015.tif'
fusai_FILE_biaozhu = '/home/lenovo/2Tdisk/Wkyao/_/biaozhu_1111.tif'

first_FILE_2017 = '/home/lenovo/2Tdisk/Wkyao/_/20170905_preliminary/preliminary/quickbird2017.tif'
first_FILE_2015 = '/home/lenovo/2Tdisk/Wkyao/_/20170905_preliminary/preliminary/quickbird2015.tif'
first_FILE_biaozhu = '/home/lenovo/2Tdisk/Wkyao/_/first_biaozhu.tif'

sec_FILE_2017 = '/home/lenovo/2Tdisk/Wkyao/_/quickbird2017_preliminary_2.tif'
sec_FILE_2015 = '/home/lenovo/2Tdisk/Wkyao/_/quickbird2015_preliminary_2.tif'
sec_FILE_biaozhu = '/home/lenovo/2Tdisk/Wkyao/_/biaozhu3000_1027.tif'

fusai_2017 = tiff.imread(fusai_FILE_2017).transpose([1, 2, 0])
fusai_2015 = tiff.imread(fusai_FILE_2015).transpose([1, 2, 0])
fusai_biaozhu = tiff.imread(fusai_FILE_biaozhu).astype(np.uint8)

first_2017 = tiff.imread(first_FILE_2017).transpose([1, 2, 0])
first_2015 = tiff.imread(first_FILE_2015).transpose([1, 2, 0])
first_biaozhu = tiff.imread(first_FILE_biaozhu).astype(np.uint8)

sec_2017 = tiff.imread(sec_FILE_2017).transpose([1, 2, 0])
sec_2015 = tiff.imread(sec_FILE_2015).transpose([1, 2, 0])
sec_biaozhu = tiff.imread(sec_FILE_biaozhu).astype(np.uint8)


def crop_first(im_biaozhu, im_2017, im_2015):
    box_size = 256
    stride = 60
    x = 0
    y = 0
    flag = 0
    a = im_2017.shape[0] - box_size
    b = im_2017.shape[1] - box_size
    print a, b

    box_biaozhu = im_biaozhu[x:x + box_size, y:y + box_size]
    box_2017_3 = im_2017[x:x + box_size, y:y + box_size, 2]
    box_2017_4 = im_2017[x:x + box_size, y:y + box_size, 3]
    box_2015_3 = im_2015[x:x + box_size, y:y + box_size, 2]
    box_2015_4 = im_2015[x:x + box_size, y:y + box_size, 3]
    while x < 3744:
        while y < 14850:
            tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/juesai/fusai_17_3/x_%d.tiff' % flag, box_2017_3)
            tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/juesai/fusai_17_4/x_%d.tiff' % flag, box_2017_4)
            tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/juesai/fusai_15_3/x_%d.tiff' % flag, box_2015_3)
            tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/juesai/fusai_15_4/x_%d.tiff' % flag, box_2015_4)
            tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/juesai/fusai_label/x_%d.tiff' % flag, box_biaozhu)
            flag = flag + 1
            print flag
            y = y + stride
            box_biaozhu = im_biaozhu[x:x + box_size, y:y + box_size]
            box_2017_3 = im_2017[x:x + box_size, y:y + box_size, 2]
            box_2017_4 = im_2017[x:x + box_size, y:y + box_size, 3]
            box_2015_3 = im_2015[x:x + box_size, y:y + box_size, 2]
            box_2015_4 = im_2015[x:x + box_size, y:y + box_size, 3]
        x = x + stride
        y = 0
        box_biaozhu = im_biaozhu[x:x + box_size, y:y + box_size]
        box_2017_3 = im_2017[x:x + box_size, y:y + box_size, 2]
        box_2017_4 = im_2017[x:x + box_size, y:y + box_size, 3]
        box_2015_3 = im_2015[x:x + box_size, y:y + box_size, 2]
        box_2015_4 = im_2015[x:x + box_size, y:y + box_size, 3]


def crop_sec(im_biaozhu, im_2017, im_2015):
    box_size = 256
    stride = 60
    x = 0
    y = 0
    flag = 0
    a = im_2017.shape[0] - box_size
    b = im_2017.shape[1] - box_size
    print a, b

    box_biaozhu = im_biaozhu[x:x + box_size, y:y + box_size]
    box_2017_3 = im_2017[x:x + box_size, y:y + box_size, 2]
    box_2017_4 = im_2017[x:x + box_size, y:y + box_size, 3]
    box_2015_3 = im_2015[x:x + box_size, y:y + box_size, 2]
    box_2015_4 = im_2015[x:x + box_size, y:y + box_size, 3]
    while x < 2744:
        while y < 14850:
            tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/juesai/2_17_3/x_%d.tiff' % flag, box_2017_3)
            tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/juesai/2_17_4/x_%d.tiff' % flag, box_2017_4)
            tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/juesai/2_15_3/x_%d.tiff' % flag, box_2015_3)
            tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/juesai/2_15_4/x_%d.tiff' % flag, box_2015_4)
            tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/juesai/2_label/x_%d.tiff' % flag, box_biaozhu)
            flag = flag + 1
            print flag
            y = y + stride
            box_biaozhu = im_biaozhu[x:x + box_size, y:y + box_size]
            box_2017_3 = im_2017[x:x + box_size, y:y + box_size, 2]
            box_2017_4 = im_2017[x:x + box_size, y:y + box_size, 3]
            box_2015_3 = im_2015[x:x + box_size, y:y + box_size, 2]
            box_2015_4 = im_2015[x:x + box_size, y:y + box_size, 3]
        x = x + stride
        y = 0
        box_biaozhu = im_biaozhu[x:x + box_size, y:y + box_size]
        box_2017_3 = im_2017[x:x + box_size, y:y + box_size, 2]
        box_2017_4 = im_2017[x:x + box_size, y:y + box_size, 3]
        box_2015_3 = im_2015[x:x + box_size, y:y + box_size, 2]
        box_2015_4 = im_2015[x:x + box_size, y:y + box_size, 3]


if __name__ == '__main__':
    crop_first(fusai_biaozhu, fusai_2017, fusai_2015)
    # crop_sec(sec_biaozhu, sec_2017, sec_2015)



