# -*- encoding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
import TensorflowUtils as utils    #加载含bn的python文件.
import read_MITSceneParsingData as scene_parsing      #输入数据格式转换的脚本read_MITSceneParsingData.py
import BatchDatsetReader as dataset      #图像预处理的,里面有resize.
from six.moves import xrange
import matplotlib.pyplot as plt
import tifffile as tiff
import scipy.misc as misc
from PIL import Image
import sys
import cv2
import misc
import crf_c as crf

import skimage, skimage.morphology, skimage.data, skimage.measure, skimage.segmentation

sys.setrecursionlimit(100000)
REGULARIZATION_RATE = 0.0001

# 添加滑动平均模型?

# 初始化变量（变量名，默认值，字符串定义）
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")  # 一个batch两张图
tf.flags.DEFINE_string("logs_dir", "logs10.13/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "enhance10.13/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")  # visualize

FILE_2017 = '/home/lenovo/2Tdisk/Wkyao/_/20170905_preliminary/preliminary/quickbird2017.tif'
im_2017 = tiff.imread(FILE_2017).transpose([1, 2, 0])  # 把通道转到最后那一维度.即新的im_2017:[,,]前两维是图像长宽,第三维是4(因为有4通道.).
FILE_2015 = '/home/lenovo/2Tdisk/Wkyao/_/20170905_preliminary/preliminary/quickbird2015.tif'
im_2015 = tiff.imread(FILE_2015).transpose([1, 2, 0])
FILE_result = '/home/lenovo/2Tdisk/Wkyao/_/2017/2017_1013.tif'
im_result = tiff.imread(FILE_result)
MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'  # 优秀的VGG19模型.
MAX_ITERATION = int(2e5 + 1)
NUM_OF_CLASSESS = 2  # 背景和新建筑.
IMAGE_RESIZE = 256


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]  # layer的类型,是conv还是relu或者pool.
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
            current = utils.batch_norm_layer(current, FLAGS.mode, scope_bn=name)    # BN处理.
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            # current = utils.max_pool_2x2(current)
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob, mean):  # keep_prob为dropout的占位符.
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)  # 下载得到VGGmodel

    #mean = model_data['normalization'][0][0][0]
    #mean_pixel = np.mean(mean, axis=(0, 1))    #[ 123.68   116.779  103.939]
    mean_pixel = mean


    weights = np.squeeze(model_data['layers'])
    processed_image = utils.process_image(image, mean_pixel)  # 图像像素值-平均像素值

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        # conv_final_layer = image_net["conv5_3"]       #这里不应该是con5_4吗?
        conv_final_layer = image_net["conv5_4"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        # w6~w8都可以做正则化的把?因为都是全连接层啊.

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")

        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)  # 全连接层才使用dropout,丢弃一些连接,使部分节点的输出为0,避免过拟合.

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")

        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        # 全连接层.  全连接层才进行正则化?
        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")

        # 对w8进行l2正则.
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        regularization = regularizer(W8)

        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")  # 扩大两倍尺寸.

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")  # 扩大两倍尺寸.

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)  # 扩大8倍尺寸.

        softmax = tf.nn.softmax(conv_t3)
        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")
    return tf.expand_dims(annotation_pred, dim=3), softmax


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)  # Adam优化算法.
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)

    return optimizer.apply_gradients(grads)


def main(argv=None):
    # 定义一下regularization:
    regularization = tf.Variable(0, dtype=tf.float32)

    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")  # dropout
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")  # 输入
    annotation = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="annotation")  # label

    mean = tf.placeholder(tf.float32, name='mean')    #给mean一个占位符.

    pred_annotation, logits = inference(image, keep_probability, mean)
    # 要对logits做一个softmax回归,就可得到预测的概率分布情况.    P75.

    loss = tf.reduce_mean(  #tf.nn.sparse_softmax_cross_entropy_with_logits()函数,第一个参数是网络前向传播不包含softmax层的计算结果.P98
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                       name="entropy"))
    trainable_var = tf.trainable_variables()  # Variable被收集在名为tf.GraphKeys.VARIABLES的colletion中
    train_op = train(loss, trainable_var)

    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)  # 数据的转换输入.
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_RESIZE}  # 将IMAGE_SIZE大小作为一个batch
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)  # 这两个是对象定义
       # 定义了train_dataset_reader
    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()  # 声明tf.train.Saver()类 用于存储模型.

    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)  # 生成check_point,下次可以从check_point继续训练
    if ckpt and ckpt.model_checkpoint_path:  # 这两行的作用是:tf.train.get_checkpoint_state()函数通过checkpoint文件(它存储所有模型的名字)找到目录下最新的模型.
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")


    if FLAGS.mode == "train":
        mymean = train_dataset_reader._read_images()  # 强行运行这个函数,把mean传过来.
       #train_dataset_reader 调用._read_images()  需要在里面修改mean是否运算和retrun.
        print ("mean:")
        print (mymean)
        mymean = [73.8613, 73.8613, 73.8613]
        for itr in xrange(MAX_ITERATION):
            print(mymean)
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85, mean: mymean}
            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss = sess.run(loss, feed_dict=feed_dict)
                print(logits.shape)
                #print(labels.shape)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                # summary_writer.add_summary(summary_str, itr)
            if itr % 500 == 0:
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        # filename = '/home/lenovo/2Tdisk/Wkyao/_/test2.jpg'
        # valid_image = misc.imread(filename)
        # valid_image = np.array([np.array([valid_image for i in range(3)])])
        # valid_image = valid_image.transpose(0, 2, 3, 1)
        # print(valid_image.shape)

        im_2017_list = []
        global im_2017  # im_2015
        for i in range(30):
            b = im_2017[0:5106, i * 500:i * 500 + 500, 3]
            b = np.array([np.array([b for i in range(3)])])
           #print(b.shape)
            b = b.transpose(0, 2, 3, 1)
            im_2017_list.append(b)
            # print (im_2017.shape)
        im_2017_list.append(
            np.array([np.array([im_2017[0:5106, 15000:15106, 3] for i in range(3)])]).transpose(0, 2, 3, 1))

        mymean = [73.9524, 73.9524, 73.9524]    #这里要改.
        allImg = []
        allImg_soft = []

        for n, im_2017_part in enumerate(im_2017_list):
            feed_dict_valid = {image: im_2017_part, keep_probability: 1.0, mean: mymean}

            #不使用crf:
            a = sess.run(pred_annotation, feed_dict=feed_dict_valid)
            print (type(a))
            a = np.mean(a, axis=(0, 3))
            allImg.append(a)


            #使用crf:
            #soft = sess.run(logits, feed_dict=feed_dict_valid)
            #运行 sess.run(logits, feed_dict=feed_dict_valid) 得到logits,即网络前向运算并softmax为概率分布后的结果.
            #soft = np.mean(soft, axis=0).transpose(2, 0, 1)
            #im_2017_mean = np.mean(im_2017_list[n], axis = 0)
            #c = crf.crf(im_2017_mean, soft)
            #allImg_soft.append(c)     #保存整张soft图.


        result = np.concatenate(tuple(allImg), axis=1).astype(np.uint8)
        #tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/2017/2017_1015.tif', result)
        #tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/2017/soft_10.16.tif', soft)


def resultImgPro(img):
    after_median = cv2.medianBlur(img, 5)
    # tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/2017_after_median.tif', after_median)
    labeled_img = skimage.measure.label(after_median)
    res = skimage.measure.regionprops(labeled_img)
    j = 0
    for i in range(len(res)):
        if res[i].area > 20000:
            # print ('del')
            j = j + 1
            # print(res[i])
    print(len(res))
    print(j)
    after_del = misc.remove_big_objects(labeled_img, 20000).astype(np.uint8)
    after_del = cv2.medianBlur(after_del, 9)
    for i in range(after_del.shape[0]):
        for j in range(after_del.shape[1]):
            if after_del[i][j] > 0:
                after_del[i][j] = 1
                # print(after_del[i][j])
    tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/2017/1/de_result_1013.tif', after_del)

def open_and_close(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 闭运算
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # 开运算
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/2017/1/open_and_close_1013.tif', img)

def change_uint8():
    IMG = im_2017_first
    result = IMG.astype(np.uint8)
    tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/20177.tif', result)

def im2017_de_im2015():
    after2017 = tiff.imread('/home/lenovo/2Tdisk/Wkyao/_/2017/2017_1012.tif')
    after2015 = tiff.imread('/home/lenovo/2Tdisk/Wkyao/_/2017/2015_1012.tif')
    result = after2017.copy()
    for n1 in range(after2015.shape[0]):
        for n2 in range(after2015.shape[1]):
            if after2015[n1][n2] == 1 and result[n1][n2] == 1:
                result[n1][n2] = 0
    resultImgPro(result)
    print(after2017.shape, after2015.shape)


if __name__ == "__main__":
    tf.app.run()
    # resultImgPro()
    # change_uint8()
    # im2017_de_im2015()
    # open_and_close(im_result)
