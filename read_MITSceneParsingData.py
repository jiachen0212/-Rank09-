# -*- encoding: utf-8 -*-
__author__ = 'charlie'
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'


def read_dataset(data_dir):     # data_dir = '/home/wkyao_check/_/'
    pickle_filename = "img.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):     # 如果在该路径没有找到file就打开url下载
        # utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
        # SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]

        result = create_image_lists(os.path.join(data_dir, 'train'))
        print os.path.join(data_dir, 'train')
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result

    return training_records, validation_records


# 可以用这个函数生成我们自己的image列表
def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    image_list = {}

    for directory in directories:   # 循环两次
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'npy')      # 原图的全路径
        print(file_glob)
        file_list.extend(glob.glob(file_glob))

        if not file_list:       # 打印了两次
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]    # 文件名
                annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')  # 标注文件
                # print annotation_file
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)    # 列表中加入字典
                else:
                    print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])   # 这个打乱没有关系
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))   # 打印出训练集的图片数量和验证集的图片数量
    return image_list