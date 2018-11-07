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
import keras
import skimage, skimage.morphology, skimage.data, skimage.measure, skimage.segmentation

sys.setrecursionlimit(100000)


# 初始化变量（变量名，默认值，字符串定义）
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")  # 一个batch两张图
tf.flags.DEFINE_string("logs_dir", "logstest/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "enhance10.13/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")  # visualize

FILE_2017 = '/home/lenovo/2Tdisk/Wkyao/_/20170905_preliminary/preliminary/quickbird2017.tif'
im_2017 = tiff.imread(FILE_2017).transpose([1, 2, 0])  # 把通道转到最后那一维度.即新的im_2017:[,,]前两维是图像长宽,第三维是4(因为有4通道.).


MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'  # 优秀的VGG19模型.

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_RESIZE = 256
#正则化系数
REGULARIZATION_RATE = 0.000001   #这个值也是需要调整的.



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
            if name[4:5] == '5':    #conv5开始使用atrous_conv2d卷积.

                # 应该是kernel_initializer,bias_initializer
                current = utils.atrous_conv2d_basic(current, kernels, bias, 2)     #rate=2,也即pad=2.

            else:
                current = utils.conv2d_basic(current, kernels, bias)
            current = utils.batch_norm_layer(current, FLAGS.mode, scope_bn=name)    # BN处理.
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            if name[4:5] == '4':
                current = utils.max_pool_1x1(current)
            else:
                current = utils.max_pool_3x3(current)
        net[name] = current

    return net


def inference(image, keep_prob, mean):  # keep_prob为dropout的占位符.
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)  # 下载得到VGGmodel
    mean_pixel = mean
    weights = np.squeeze(model_data['layers'])    #weights初始化为vgg网络的权值.

    processed_image = utils.process_image(image, mean_pixel)  # 图像像素值-平均像素值
    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_4"]

        pool5 = utils.max_pool_1x1(conv_final_layer)

        # 7x7改成3x3，4096改成了1024，可能特征不够？

        W6 = utils.weight_variable([3, 3, 512, 1024], name="W6")
        b6 = utils.bias_variable([1024], name="b6")

        # 如果报错 把W6，b6改为kernel_initializer=，bias_initializer=
        # data_format = "channels_last" 为默认
        #使用不同rate的孔卷积.
        Fc6_1 = utils.atrous_conv2d_basic(pool5, W6, b6, 6)
        Fc6_2 = utils.atrous_conv2d_basic(pool5, W6, b6, 12)
        Fc6_3 = utils.atrous_conv2d_basic(pool5, W6, b6, 18)
        Fc6_4 = utils.atrous_conv2d_basic(pool5, W6, b6, 24)

        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        #regularization1 = regularizer(W6)

        relu6_1 = tf.nn.relu(Fc6_1, name="relu6_1")
        relu6_2 = tf.nn.relu(Fc6_2, name="relu6_2")
        relu6_3 = tf.nn.relu(Fc6_3, name="relu6_3")
        relu6_4 = tf.nn.relu(Fc6_4, name="relu6_4")

        relu_dropout6_1 = tf.nn.dropout(relu6_1, keep_prob=keep_prob)
        relu_dropout6_2 = tf.nn.dropout(relu6_2, keep_prob=keep_prob)
        relu_dropout6_3 = tf.nn.dropout(relu6_3, keep_prob=keep_prob)
        relu_dropout6_4 = tf.nn.dropout(relu6_4, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 1024, 1024], name="W7")
        b7 = utils.bias_variable([1024], name="b7")


        Fc7_1 = utils.conv2d_basic(relu_dropout6_1, W7, b7)
        Fc7_2 = utils.conv2d_basic(relu_dropout6_2, W7, b7)
        Fc7_3 = utils.conv2d_basic(relu_dropout6_3, W7, b7)
        Fc7_4 = utils.conv2d_basic(relu_dropout6_4, W7, b7)
        #regularization2 = regularizer(W7)

        relu7_1 = tf.nn.relu(Fc7_1, name="relu7_1")
        relu7_2 = tf.nn.relu(Fc7_2, name="relu7_2")
        relu7_3 = tf.nn.relu(Fc7_3, name="relu7_3")
        relu7_4 = tf.nn.relu(Fc7_4, name="relu7_4")

        relu_dropout7_1 = tf.nn.dropout(relu7_1, keep_prob=keep_prob)
        relu_dropout7_2 = tf.nn.dropout(relu7_2, keep_prob=keep_prob)
        relu_dropout7_3 = tf.nn.dropout(relu7_3, keep_prob=keep_prob)
        relu_dropout7_4 = tf.nn.dropout(relu7_4, keep_prob=keep_prob)


        W8 = utils.weight_variable([1, 1, 1024, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")


        Fc8_1 = utils.conv2d_basic(relu_dropout7_1, W8, b8)
        Fc8_2 = utils.conv2d_basic(relu_dropout7_2, W8, b8)
        Fc8_3 = utils.conv2d_basic(relu_dropout7_3, W8, b8)
        Fc8_4 = utils.conv2d_basic(relu_dropout7_4, W8, b8)
        Fc8 = tf.add_n([Fc8_1, Fc8_2, Fc8_3, Fc8_4], name="Fc8")    #F8的各个层尺寸一样,感受野不同.
        #regularization3 = regularizer(W8)

        # w6-w7 L2正则化.
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(W6))
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(W7))
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(W8))


        shape = tf.shape(image)
        resize_Fc8 = tf.image.resize_images(Fc8, (shape[1], shape[2]))  #tf自带的扩尺寸函数resize_images(),默认双线性插值.尺寸扩大8倍至原尺寸256x256
        softmax = tf.nn.softmax(resize_Fc8)    # tf.nn.softmax(),使前向计算结果转为概率分布
        #annotation_pred = tf.argmax(Fc8, dimension=3, name="prediction")
        annotation_pred = tf.argmax(softmax, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), resize_Fc8


def train(loss_val, var_list):

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)  # Adam优化算法.
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    '''
    #定义一下regularization:
    regularization = tf.Variable(0, dtype=tf.float32)
    '''
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")   #dropout
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="annotation")
    mean = tf.placeholder(tf.float32, name='mean')     #给mean一个占位符.
    pred_annotation, logits = inference(image, keep_probability, mean)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                       name="entropy"))
    tf.add_to_collection('losses', loss)
    loss = tf.add_n(tf.get_collection('losses'))    #loss为总的正则化+weights之和.

    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)  # 数据的转换输入.
    print(len(train_records))
    print(len(valid_records))

    trainable_var = tf.trainable_variables()  # Variable被收集在名为tf.GraphKeys.VARIABLES的colletion中

    train_op = train(loss, trainable_var)

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_RESIZE}  # 将IMAGE_SIZE大小作为一个batch
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)  # 这两个是对象定义

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()


    # 这里的初始化方式可能可以修改以改善结果.
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)  # 生成check_point,下次可以从check_point继续训练
    if ckpt and ckpt.model_checkpoint_path:  # 这两行的作用是:tf.train.get_checkpoint_state()函数通过checkpoint文件(它存储所有模型的名字)找到目录下最新的模型.
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        #mymean = train_dataset_reader._read_images()   #强行运行这个函数,把mean传过来.
        #train_dataset_reader 调用._read_images()  需要在里面修改mean是否运算和retrun.
        #生成本次数据集的mean值.
        #print("mean:")
        #print(mymean)
        mymean = [73.8613, 73.8613, 73.8613]     #这里要根据数据集更新.

        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85, mean: mymean}
            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss = sess.run(loss, feed_dict=feed_dict)
                # print()
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                # summary_writer.add_summary(summary_str, itr)
            if itr % 500 == 0:
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        im_2017_list = []
        global im_2017
        for i in range(30):
            b = im_2017[0:5106, i * 500:i * 500 + 500, 3]
            b = np.array([np.array([b for i in range(3)])])
            b = b.transpose(0, 2, 3, 1)
            im_2017_list.append(b)
        im_2017_list.append(
            np.array([np.array([im_2017[0:5106, 15000:15106, 3] for i in range(3)])]).transpose(0, 2, 3, 1))

        allImg = []
        mymean = [73.8613, 73.8613, 73.8613]   #这里要根据数据集更新.

        for n, im_2017_part in enumerate(im_2017_list):
            feed_dict_valid = {image: im_2017_part, keep_probability: 1.0, mean:mymean}
            a = sess.run(pred_annotation, feed_dict=feed_dict_valid)
            a = np.mean(a, axis=(0, 3))
            allImg.append(a)

        result = np.concatenate(tuple(allImg), axis=1).astype(np.uint8)
        tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/2017/deep_1015_1.tif', result)



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
    tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/2017/de_result_1012.tif', after_del)

def open_and_close(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 闭运算
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # 开运算
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/2017/open_and_close_1013.tif', img)

def change_uint8(img):
    IMG = img
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
