# -*- encoding: utf-8 -*-
"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
import tensorflow as tf
import cv2


class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        self.__channels = True
        self.images = np.array([self._transform1(filename['image']) for filename in self.files])

        print ("images's:"),
        print self.images.shape

        self.__channels = False
        self.annotations = np.array(
            [np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files])
        print ('annotations')
        print (self.annotations.shape)

        '''
        我觉得用这段代码换下面这段代码会快很多...
        '''
        # print self.annotations.shape
        for n in range(self.annotations.shape[0]):  # self.annotations.shape[0]:图像总数量.
            # (10240, 256, 256, 1) numpy not tuple
            # print (self.annotations.shape)
            self.annotations[n] = self.annotations[n].astype(np.bool)  # 转成pool型,则所有大于等于1的数全部变成true.
            self.annotations[n] = self.annotations[n].astype(np.uint8)  # true转uint8直接变成1.

        # for n1 in range(150):   # test
        #    for n2 in range(150):
        #        print self.annotations[0][n1][n2][0]

    def _transform1(self, filename):
        image = np.load(filename)    #以.npy格式读取图像.
        #print '******'
        if image.shape == 3:
            image = tf.squeeze(image, squeeze_dims=[2])
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def _transform(self, filename):
        image = misc.imread(filename)
        # image = cv2.imread(filename, -1)
        # print (image.shape)
        if image.shape == 3:
            image = tf.squeeze(image, squeeze_dims=[2])
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:  # image.shape[0] == images num
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]
