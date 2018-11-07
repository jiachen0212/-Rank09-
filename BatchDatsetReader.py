# -*- encoding: utf:8 -*-
"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
import tensorflow as tf


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
        self.images = np.array([np.expand_dims(self._transform1(filename['image']), axis=3) for filename in self.files])
        print ("images's shape:"),
        print self.images.shape
        print '*****'
        # mymean = np.mean(self.images, axis=(0,1,2))   #images:[12800,256,256,3]压缩前三维,得到[1*3]mean像素.
        # print mymean
        self.__channels = False
        self.annotations = np.array([np.expand_dims(self._transform2(filename['annotation']), axis=3) for filename in self.files])
        print self.images.shape
        print self.annotations.shape

        # print self.annotations.shape
        for n in range(self.annotations.shape[0]):    #self.annotations.shape[0]:图像总数量.
            # (10240, 256, 256, 1) numpy not tuple
            # print (self.annotations.shape)
            self.annotations[n] = self.annotations[n].astype(np.bool)   #转成pool型,则所有大于等于1的数全部变成true.
            self.annotations[n] = self.annotations[n].astype(np.uint8)  #true转uint8直接变成1.


    def _transform(self, filename):
        image = misc.imread(filename)
        print(image.shape)
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


    def _transform1(self, filename):
        # image = misc.imread(filename)
        image = np.load(filename)
        # image = cv2.imread(filename, -1)
        # print image.shape

        # if image.shape == 3:
        #     image = tf.squeeze(image, squeeze_dims=[2])
        # if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
        #     image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image, [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def _transform2(self, filename):
        image = misc.imread(filename)
        # image = np.load(filename)
        # image = cv2.imread(filename, -1)
        # print image.shape

        # if image.shape == 3:
        #     image = tf.squeeze(image, squeeze_dims=[2])
        # if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
        #     image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image, [resize_size, resize_size], interp='nearest')
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
        if self.batch_offset > self.images.shape[0]:    # image.shape[0] == images num
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
