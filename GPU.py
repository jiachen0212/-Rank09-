# -*- encoding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
import TensorflowUtils as utils  # 加载含bn的python文件.
import read_MITSceneParsingData as scene_parsing  # 输入数据格式转换的脚本read_MITSceneParsingData.py
import BatchDatsetReader1 as dataset  # 图像预处理的,里面有resize.
from six.moves import xrange
import tifffile as tiff
import cv2
import datetime

REGULARIZATION_RATE = 0.0001

# 初始化变量（变量名，默认值，字符串定义）
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "4", "batch size for training")  # 一个batch两张图
tf.flags.DEFINE_string("logs_dir", "logs_deep1025_gpu/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "enhance10.24/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")  # visualize

im_2017 = cv2.imread('/home/lenovo/2Tdisk/Wkyao/_/20170905_preliminary/preliminary/2017.jpg')
print(im_2017.shape)
FILE_2017 = '/home/lenovo/2Tdisk/Wkyao/_/20170905_preliminary/preliminary/quickbird2017.tif'
im_2017tif = tiff.imread(FILE_2017).transpose([1, 2, 0])  # 把通道转到最后那一维度.即新的im_2017:[,,]前两维是图像长宽,第三维是4(因为有4通道.).
FILE_2015 = '/home/lenovo/2Tdisk/Wkyao/_/20170905_preliminary/preliminary/quickbird2015.tif'
im_2015 = tiff.imread(FILE_2015).transpose([1, 2, 0])
FILE_result = '/home/lenovo/2Tdisk/Wkyao/_/2017/deep_10172216.tif'
im_result = tiff.imread(FILE_result)

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'  # 优秀的VGG19模型.

MAX_ITERATION = int(2e4 + 1)
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
                # 应该是kernel_initializer,bias_initializer
                current = utils.atrous_conv2d_basic(current, kernels, bias, 2)  # rate=2,也即pad=2.
                current = utils.batch_norm_layer(current, FLAGS.mode, scope_bn=name)  # BN处理.
            else:  # conv1-4
                current = utils.conv2d_basic(current, kernels, bias)
                current = utils.batch_norm_layer(current, FLAGS.mode, scope_bn=name)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            if name[4:5] == '4':
                current = utils.max_pool_1x1(current)
            else:
                current = utils.max_pool_3x3(current)
        net[name] = current
    return net

# tf.glorot_uniform_initializer()

def inference(image, keep_prob, mean):  # keep_prob为dropout的占位符.
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)  # 下载得到VGGmodel
    mean_pixel = mean
    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)  # 图像像素值-平均像素值
    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_4"]

        pool5 = utils.max_pool_1x1(conv_final_layer)

        W6 = utils.weight_variable([3, 3, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")

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

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")

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

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")

        Fc8_1 = utils.conv2d_basic(relu_dropout7_1, W8, b8)
        Fc8_2 = utils.conv2d_basic(relu_dropout7_2, W8, b8)
        Fc8_3 = utils.conv2d_basic(relu_dropout7_3, W8, b8)
        Fc8_4 = utils.conv2d_basic(relu_dropout7_4, W8, b8)

        Fc8 = tf.add_n([Fc8_1, Fc8_2, Fc8_3, Fc8_4], name="Fc8")  # F8的各个层尺寸一样,感受野不同.

        shape = tf.shape(image)
        resize_Fc8 = tf.image.resize_images(Fc8, (shape[1], shape[2]))  # 线性插值.尺寸扩大8倍至原尺寸256x256
        softmax = tf.nn.softmax(resize_Fc8)  # tf.nn.softmax(),使前向计算结果转为概率分布
        annotation_pred = tf.argmax(softmax, dimension=3, name="prediction")
    return tf.expand_dims(annotation_pred, dim=3), resize_Fc8, softmax


def tower_loss(loss, scope):
    tf.add_to_collection('losses', loss)
    losses = tf.get_collection('losses', scope=scope)
    total_loss = tf.add_n(tf.get_collection('losses', scope=scope), name='total_loss')
    return total_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train(loss):
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
    tower_grads = []
    for i in range(2):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                lossss = tower_loss(loss, scope)
                tf.get_variable_scope().reuse_variables()
                grads = opt.compute_gradients(lossss)
                tower_grads.append(grads)
    grads = average_gradients(tower_grads)  # 求两个gpu计算的平均梯度
    # apply_gradient_op = opt.apply_gradients(grads)
    return opt.apply_gradients(grads)


def main(argv=None):
    #是否需要申明把简单的计算放在cpu下.
    regularization = tf.Variable(0, dtype=tf.float32)

    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")  # dropout
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")  # 输入
    annotation = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="annotation")  # label
    mean = tf.placeholder(tf.float32, name='mean')  # 给mean一个占位符.
    pred_annotation, logits, softmax = inference(image, keep_probability, mean)  # logits=resize_F8

    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                       name="entropy"))
    train_op = train(loss)
    tf.summary.scalar("entropy", loss)  # train val公用一个loss节点运算.


    print("Setting up summary op...")

    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)  # 数据的转换输入.
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_RESIZE}  # 将IMAGE_SIZE大小作为一个batch

    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)  # 是train模式也把validation载入进来.

    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver()  # 声明tf.train.Saver()类 用于存储模型.
    summary_op = tf.summary.merge_all()  # 汇总所有summary.
    summary_writer_train = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    summary_writer_val = tf.summary.FileWriter(FLAGS.logs_dir + '/val')  # 这里不需要再加入graph.

    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)  # 生成check_point,下次可以从check_point继续训练
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":

        mymean = [73.9524, 73.9524, 73.9524]  # 这里要改.
        for itr in xrange(MAX_ITERATION):  # 修改itr数值,接着之前的模型数量后继续训练.
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)

            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85, mean: mymean}
            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)

                summary_writer_train.add_summary(summary_str, itr)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))

            if itr % 100 == 0:  # 每训500次测试一下验证集.

                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss, summary_str = sess.run([loss, summary_op], feed_dict={image: valid_images,
                                                                                    annotation: valid_annotations,
                                                                                    keep_probability: 1.0,
                                                                                    mean: mymean})  # 计算新节点loss_valid的值.
                summary_writer_val.add_summary(summary_str, itr)
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
        summary_writer_val.close()
        summary_writer_train.close()

    elif FLAGS.mode == "test":
        im_2017_list = []
        global im_2017  # [5106, 15106, 3]
        global im_2017tif
        for i in range(50):
            b = im_2017tif[0:5106, i * 300:i * 300 + 300, 3]
            b = np.array([np.array([b for i in range(3)])])
            b = b.transpose(0, 2, 3, 1)
            im_2017_list.append(b)
        im_2017_list.append(
            np.array([np.array([im_2017tif[0:5106, 15000:15106, 3] for i in range(3)])]).transpose(0, 2, 3, 1))

        allImg = []
        mymean = [73.9524, 73.9524, 73.9524]

        for n, im_2017_part in enumerate(im_2017_list):
            feed_dict_test = {image: im_2017_part, keep_probability: 1.0, mean: mymean}
            a = sess.run(pred_annotation, feed_dict=feed_dict_test)
            a = np.mean(a, axis=(0, 3))
            allImg.append(a)
        result = np.concatenate(tuple(allImg), axis=1).astype(np.uint8)
        tiff.imsave('/home/lenovo/2Tdisk/Wkyao/_/2017/deep_1025_2w.tif', result)


if __name__ == "__main__":
    tf.app.run()
    # resultImgPro()
    # change_uint8()
    # im2017_de_im2015()
    # open_and_close(im_result)