# -*- encoding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
import TensorflowUtils as utils                     # 加载含bn的python文件.
import read_MITSceneParsingData as scene_parsing    # 输入数据格式转换的脚本read_MITSceneParsingData.py
# import BatchDatsetReader1 as dataset     # 读.png!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import BatchNpReader as dataset
from six.moves import xrange
import tifffile as tiff
import scipy.misc as misc
import cv2
import misc
import skimage, skimage.morphology, skimage.data, skimage.measure, skimage.segmentation
import crf_c as crf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
REGULARIZATION_RATE = 0.0001

# 初始化变量（变量名，默认值，字符串定义）
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "8", "batch size for training")  # 一个batch两张图
tf.flags.DEFINE_string("logs_dir", "/home/lenovo/256Gdisk/tainchi/log1120_4/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "/home/lenovo/256Gdisk/juesai/enhance_4/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "test", "Mode train/ test/ visualize")  # visualize


FILE_new_2017 = '/home/lenovo/2Tdisk/Wkyao/_/20171105_quarterfinals/quarterfinals_2017.tif'
new_2017 = tiff.imread(FILE_new_2017).transpose([1, 2, 0])

FILE_2017 = '/home/lenovo/2Tdisk/Wkyao/_/20170905_preliminary/preliminary/quickbird2017.tif'
im_2017tif = tiff.imread(FILE_2017).transpose([1, 2, 0])  # 把通道转到最后那一维度.即新的im_2017:[,,]前两维是图像长宽,第三维是4(因为有4通道.).
FILE_2015 = '/home/lenovo/2Tdisk/Wkyao/_/20170905_preliminary/preliminary/quickbird2015.tif'
im_2015 = tiff.imread(FILE_2015).transpose([1, 2, 0])
FILE_result = '/home/lenovo/2Tdisk/Wkyao/_/2017/deep_10172216.tif'
im_result = tiff.imread(FILE_result)
FILE_biaozhu = '/home/lenovo/2Tdisk/Wkyao/_/biaozhu_1110.tif'    #fusai标注文件
label = tiff.imread(FILE_biaozhu)

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'  # 优秀的VGG19模型.

MAX_ITERATION = int(2e3 + 1)
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

            if name[4:5] == '5':  # conv5开始使用atrous_conv2d卷积.
                current = utils.atrous_conv2d_basic(current, kernels, bias, 2)  # rate=2,也即pad=2.
                current = utils.batch_norm_layer(current, FLAGS.mode, scope_bn=name)  # BN处理.
            else:  # conv1-4
                current = utils.conv2d_basic(current, kernels, bias)
                current = utils.batch_norm_layer(current, FLAGS.mode, scope_bn=name)  # BN处理.
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            if name[4:5] == '4':
                current = utils.max_pool_1x1(current)
            else:
                current = utils.max_pool_3x3(current)
        net[name] = current

    return net


def inference(image, keep_prob):  # keep_prob为dropout的占位符.
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)  # 下载得到VGGmodel
    # mean_pixel = mean
    weights = np.squeeze(model_data['layers'])

    # processed_image = utils.process_image(image, mean_pixel)  # 图像像素值-平均像素值
    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, image)
        conv_final_layer = image_net["conv5_4"]

        pool5 = utils.max_pool_1x1(conv_final_layer)

        # w6~w8都可以做正则化的把?因为都是全连接层啊.
        # 7x7改成3x3，4096改成了1024，可能特征不够？

        # 新加的w6-w8 b6-b8都自带初始化.
        W6 = utils.weight_variable([3, 3, 512, 512], name="W6")
        b6 = utils.bias_variable([512], name="b6")

        # data_format = "channels_last" 为默认
        # 使用不同rate的孔卷积.
        Fc6_1 = utils.atrous_conv2d_basic(pool5, W6, b6, 6)
        Fc6_2 = utils.atrous_conv2d_basic(pool5, W6, b6, 12)
        Fc6_3 = utils.atrous_conv2d_basic(pool5, W6, b6, 18)
        Fc6_4 = utils.atrous_conv2d_basic(pool5, W6, b6, 24)

        Bn6_1 = utils.batch_norm_layer(Fc6_1, FLAGS.mode, scope_bn='Bn')  # bn处理要在relu之前.
        Bn6_2 = utils.batch_norm_layer(Fc6_2, FLAGS.mode, scope_bn='Bn')
        Bn6_3 = utils.batch_norm_layer(Fc6_3, FLAGS.mode, scope_bn='Bn')
        Bn6_4 = utils.batch_norm_layer(Fc6_4, FLAGS.mode, scope_bn='Bn')

        relu6_1 = tf.nn.relu(Bn6_1, name="relu6_1")
        relu6_2 = tf.nn.relu(Bn6_2, name="relu6_2")
        relu6_3 = tf.nn.relu(Bn6_3, name="relu6_3")
        relu6_4 = tf.nn.relu(Bn6_4, name="relu6_4")

        relu_dropout6_1 = tf.nn.dropout(relu6_1, keep_prob=keep_prob)
        relu_dropout6_2 = tf.nn.dropout(relu6_2, keep_prob=keep_prob)
        relu_dropout6_3 = tf.nn.dropout(relu6_3, keep_prob=keep_prob)
        relu_dropout6_4 = tf.nn.dropout(relu6_4, keep_prob=keep_prob)

        '''
        # 原来的代码
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)  # 全连接层才使用dropout,丢弃一些连接,使部分节点的输出为0,避免过拟合.
        '''

        W7 = utils.weight_variable([1, 1, 512, 512], name="W7")
        b7 = utils.bias_variable([512], name="b7")

        Fc7_1 = utils.conv2d_basic(relu_dropout6_1, W7, b7)
        Fc7_2 = utils.conv2d_basic(relu_dropout6_2, W7, b7)
        Fc7_3 = utils.conv2d_basic(relu_dropout6_3, W7, b7)
        Fc7_4 = utils.conv2d_basic(relu_dropout6_4, W7, b7)

        Bn7_1 = utils.batch_norm_layer(Fc7_1, FLAGS.mode, scope_bn='Bn')
        Bn7_2 = utils.batch_norm_layer(Fc7_2, FLAGS.mode, scope_bn='Bn')
        Bn7_3 = utils.batch_norm_layer(Fc7_3, FLAGS.mode, scope_bn='Bn')
        Bn7_4 = utils.batch_norm_layer(Fc7_4, FLAGS.mode, scope_bn='Bn')

        relu7_1 = tf.nn.relu(Bn7_1, name="relu7_1")
        relu7_2 = tf.nn.relu(Bn7_2, name="relu7_2")
        relu7_3 = tf.nn.relu(Bn7_3, name="relu7_3")
        relu7_4 = tf.nn.relu(Bn7_4, name="relu7_4")

        relu_dropout7_1 = tf.nn.dropout(relu7_1, keep_prob=keep_prob)
        relu_dropout7_2 = tf.nn.dropout(relu7_2, keep_prob=keep_prob)
        relu_dropout7_3 = tf.nn.dropout(relu7_3, keep_prob=keep_prob)
        relu_dropout7_4 = tf.nn.dropout(relu7_4, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 512, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")

        Fc8_1 = utils.conv2d_basic(relu_dropout7_1, W8, b8)
        Fc8_2 = utils.conv2d_basic(relu_dropout7_2, W8, b8)
        Fc8_3 = utils.conv2d_basic(relu_dropout7_3, W8, b8)
        Fc8_4 = utils.conv2d_basic(relu_dropout7_4, W8, b8)


        Fc8 = tf.add_n([Fc8_1, Fc8_2, Fc8_3, Fc8_4], name="Fc8")  # F8的各个层尺寸一样,感受野不同.

        shape = tf.shape(image)
        # print (shape[1], shape[2], '*****************')
        resize_Fc8 = tf.image.resize_images(Fc8,
                                            (shape[1], shape[2]))  # tf自带的扩尺寸函数resize_images(),默认双线性插值.尺寸扩大8倍至原尺寸256x256
        softmax = tf.nn.softmax(resize_Fc8)  # tf.nn.softmax(),使前向计算结果转为概率分布
        annotation_pred = tf.argmax(softmax, dimension=3, name="prediction")
    return tf.expand_dims(annotation_pred, dim=3), resize_Fc8, softmax


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)  # Adam优化算法.
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)  # 计算梯度,返回权重更新
    return optimizer.apply_gradients(grads)
    # return loss_val


def main(argv=None):
    # 定义一下regularization:
    regularization = tf.Variable(0, dtype=tf.float32)

    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")  # dropout
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")  # 输入
    annotation = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="annotation")  # label
    # mean = tf.placeholder(tf.float32, name='mean')
    pred_annotation, logits, softmax = inference(image, keep_probability)  # logits=resize_F8

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                       # tf.nn.sparse_softmax_cross_entropy_with_logits自动对logits(resize_F8)做了softmax处理.
                                                       labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                       name="entropy"))
    tf.summary.scalar("entropy", loss)  # train val公用一个loss节点运算.

    trainable_var = tf.trainable_variables()  # Variable被收集在名为tf.GraphKeys.VARIABLES的colletion中

    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)  # 数据的转换输入.
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_RESIZE}  # 将IMAGE_SIZE大小作为一个batch

    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)

    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver()  # 声明tf.train.Saver()类 用于存储模型.
    summary_op = tf.summary.merge_all()  # 汇总所有summary.
    summary_writer_train = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    summary_writer_val = tf.summary.FileWriter(FLAGS.logs_dir + '/val')  # 这里不需要再加入graph.

    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)  # 生成check_point,下次可以从check_point继续训练
    if ckpt and ckpt.model_checkpoint_path:  # 这两行的作用是:tf.train.get_checkpoint_state()函数通过checkpoint文件(它存储所有模型的名字)找到目录下最新的模型.
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        # mymean = train_dataset_reader._read_images()   #强行运行这个函数,把mean传过来.
        # train_dataset_reader 调用._read_images()  需要在里面修改mean是否运算和return.
        # 生成本次数据集的mean值.
        # mymean = [42.11049008, 65.75782253, 74.11216841]    #这里要根据数据集更新.
        for itr in xrange(MAX_ITERATION):  # 修改itr数值,接着之前的模型数量后继续训练.
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}
            sess.run(train_op, feed_dict=feed_dict)
            if itr % 10 == 0:
                # 这里不要运算loss
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                summary_writer_train.add_summary(summary_str, itr)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))

            if itr % 100 == 0:  # 每训100次测试一下验证集.
                # valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                # valid_loss, summary_str = sess.run([loss, summary_op], feed_dict={image: valid_images,
                #                                                                          annotation: valid_annotations,
                #                                                                          keep_probability: 1.0,
                #                                                                          mean: mymean})    #计算新节点loss_valid的值.
                # summary_writer_val.add_summary(summary_str, itr)
                # print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
            # if itr % 1000 == 0:
            #     im_2017_list = []
            #     allImg = []
            #     im_2017_list = ge2017()
            #     for n, im_2017_part in enumerate(im_2017_list):
            #         # print(im_2017_part.shape)
            #         feed_dict_test = {image: im_2017_part, keep_probability: 1.0, mean: mymean}
            #         a = sess.run(pred_annotation, feed_dict=feed_dict_test)
            #         a = np.mean(a, axis=(0, 3))
            #         allImg.append(a)
            #
            #     Img = np.concatenate(tuple(allImg), axis=1)  # axis = 1 在第二个维度上进行拼接.
            #     f1socre = f1.calf1(label, Img)
            #     print("%s ---> F1: %f" % (datetime.datetime.now(), f1socre))
        # summary_writer_val.close()
        summary_writer_train.close()

    elif FLAGS.mode == "visualize":
        mymean = [73.9524, 73.9524, 73.9524]

        validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})

        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5 + itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5 + itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5 + itr))
            print("Saved image: %d" % itr)

    elif FLAGS.mode == "test":
        new_2017 = np.load('/home/lenovo/256Gdisk/tainchi/juesaidata/test.npy')
        im_2017_list = []
        im_2017_list2 = []
        b = []
        global im_2017  # [5106, 15106, 3]
        #global new_2017  # [8106, 15106, 3]
        global new_2015
        for i in range(25):
            #b = new_2017[0:4000, i * 600:i * 600 + 600, 2]   #训练第三通道的话,测试的时候也要换成第三通道.
            b = new_2017[0:3000, i * 600:i * 600 + 600]  # 训练第三通道的话,测试的时候也要换成第三通道.
            b = np.array([np.array([b for i in range(3)])])
            b = b.transpose(0, 2, 3, 1)
            im_2017_list.append(b)
            print(b.shape)

            '''
            #　多通道
            b = im_2017[0:5106, i * 300 : i * 300 + 300, :]
            b = np.array([b])
            # print(b.shape)
            # b = b.transpose(0, 2, 3, 1)
            im_2017_list.append(b)
        # im_2017_list.append(np.array([np.array([im_2017[0:5106, 15000:15106, 3] for i in range(3)])]).transpose(0, 2, 3, 1))
        #im_2017_list.append(np.array([im_2017[0:5106, 15000:15106, :]]))     # .transpose(0, 2, 3, 1))
        im_2017_list.append(np.array([im_2017[0:5106, 15000:15106, :]]))
        '''

        im_2017_list.append(
            np.array([np.array([new_2017[0:3000, 15000:15106] for i in range(3)])]).transpose(0, 2, 3, 1))
        #im_2017_list.append(np.array([np.array([new_2017[0:4000, 15000:15106]])]))
        '''
        for i in range(50)
            b = new_2017[5106:8106, i * 300:i * 300 + 300, 3]
            b = np.array([np.array([b for i in range(3)])])
            b = b.transpose(0, 2, 3, 1)
            im_2017_list2.append(b)
        im_2017_list2.append(
            np.array([np.array([new_2017[5106:8106, 15000:15106, 3] for i in range(3)])]).transpose(0, 2, 3, 1))
        '''
        allImg = []
        allImg2 = []
        allImg_soft = []
        allImg_crf = []
        #mymean = [73.9524, 73.9524, 73.9524]

        for n, im_2017_part in enumerate(im_2017_list):
            # print(im_2017_part.shape)
            feed_dict_test = {image: im_2017_part, keep_probability: 1.0}
            a = sess.run(pred_annotation, feed_dict=feed_dict_test)
            a = np.mean(a, axis=(0, 3))
            allImg.append(a)

            # 使用crf:
            soft = sess.run(softmax, feed_dict=feed_dict_test)
            # 运行 sess.run(softmax, feed_dict=feed_dict_test) 得到softmax,即网络前向运算并softmax为概率分布后的结果.
            soft = np.mean(soft, axis=0).transpose(2, 0, 1)
            # soft = soft.transpose(2, 0, 1)

            im_2017_mean = np.mean(im_2017_list[n], axis=0)

            # print (im_2017_mean.shape)     #(5106, 300, 3)
            c = crf.crf(im_2017_mean, soft)
            # print (c.shape)    #(5106, 300)
            allImg_crf.append(c)
            allImg_soft.append(soft)  # 保存整张soft图.
        Crf = np.concatenate(tuple(allImg_crf), axis=1)  # axis = 1 在第二个维度上进行拼接.
        softmax = np.concatenate(tuple(allImg_soft), axis=2)
        tiff.imsave('/home/lenovo/256Gdisk/tainchi/vgg/fo_crf_1120_4.tif', Crf)
        np.save('/home/lenovo/256Gdisk/tainchi/vgg/fo_1120_4.npy', softmax)
        res1 = np.concatenate(tuple(allImg), axis=1).astype(np.uint8)
        tiff.imsave('/home/lenovo/256Gdisk/tainchi/vgg/fo_1120_4.tif', res1)
        open_and_close(Crf)     #膨胀操作.


def open_and_close(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    # 闭运算
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # 开运算
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # 膨胀
    img = cv2.dilate(img, kernel)
    tiff.imsave('/home/lenovo/256Gdisk/tainchi/vgg/fo_open_1120_4.tif', img)

def ge2017():
    im_2017_list = []
    im_2017_list2 = []
    b = []
    global im_2017  # [5106, 15106, 3]
    global new_2017  # [8106, 15106, 3]
    global new_2015
    for i in range(25):
        b = new_2017[0:3000, i * 600:i * 600 + 600, 2]  # 训练第三通道的话,测试的时候也要换成第三通道.
        b = np.array([np.array([b for i in range(3)])])
        b = b.transpose(0, 2, 3, 1)
        im_2017_list.append(b)
        print(b.shape)
    im_2017_list.append(
        np.array([np.array([new_2017[0:3000, 15000:15106, 2] for i in range(3)])]).transpose(0, 2, 3, 1))
    return im_2017_list

def change_uint8(img):
    IMG = img
    result = IMG.astype(np.uint8)
    tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/2017.tif', result)


# def im2017_de_im2015():
#     after2017 = tiff.imread('/home/lenovo/2Tdisk/Wkyao/_/2017/2017_1012.tif')
#     after2015 = tiff.imread('/home/lenovo/2Tdisk/Wkyao/_/2017/2015_1012.tif')
#     result = after2017.copy()
#     for n1 in range(after2015.shape[0]):
#         for n2 in range(after2015.shape[1]):
#             if after2015[n1][n2] == 1 and result[n1][n2] == 1:
#                 result[n1][n2] = 0
#     resultImgPro(result)
#     print(after2017.shape, after2015.shape)


if __name__ == "__main__":
    tf.app.run()
    # resultImgPro()
    # change_uint8()
    # im2017_de_im2015()
    # open_and_close()
