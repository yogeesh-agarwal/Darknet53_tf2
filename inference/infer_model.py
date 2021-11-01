import tensorflow as tf
import numpy as np
import pickle

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, conv_weights , bn_weights , kernel_size = None , downsample = False):
        super().__init__()
        def get_pad_info(kernel_size):
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            return pad_beg , pad_end

        self.strides = 1
        self.padding = "SAME"
        if downsample:
            self.strides = 2
            self.padding = "VALID"
            self.pad_beg , self.pad_end = get_pad_info(kernel_size)

        self.conv_weights = conv_weights
        self.beta = bn_weights["beta"]
        self.mean = bn_weights["mean"]
        self.gamma = bn_weights["gamma"]
        self.variance = bn_weights["variance"]
        self.bn_layer = tf.keras.layers.BatchNormalization()

    def call(self , input , training = False):
        conv_out = tf.nn.conv2d(input , filters = self.conv_weights , strides = [1 , self.strides , self.strides , 1] , padding = self.padding)
        bn_out = tf.nn.batch_normalization(conv_out , self.mean , self.variance , self.beta , self.gamma , 1e-03)
        lr_out = tf.nn.leaky_relu(bn_out , alpha = 0.1)
        return lr_out

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self , conv_weights , bn_weights):
        super().__init__()
        self.residual_block1 = ConvBlock(conv_weights[0] , bn_weights[0])
        self.residual_block2 = ConvBlock(conv_weights[1] , bn_weights[1])
        self.block_out = tf.keras.layers.Add()

    def call(self , input , training = False):
        shortcut = input
        residual_b1 = self.residual_block1(input , training = training)
        residual_b2 = self.residual_block2(residual_b1 , training = training)
        blk_out = self.block_out([residual_b2 , shortcut])
        return blk_out

class Darknet53Infer(tf.keras.Model):
    def __init__(self , darknet53_weights_file , darknet53_bn_file , dense_weights_file):
        super().__init__(name = "darknet53_infer")
        self.darknet53_weights = self.load_pickle(darknet53_weights_file)
        self.dense_weights = self.load_pickle(dense_weights_file)
        self.darknet53_bn = self.load_pickle(darknet53_bn_file)
        self.conv_index = 0
        self.bn_index = 0

        self.conv_b1 = ConvBlock(conv_weights = self.get_weights() , bn_weights = self.get_bn_weights())
        self.conv_b2 = ConvBlock(conv_weights = self.get_weights() , bn_weights = self.get_bn_weights() , kernel_size = 3, downsample = True)
        self.rb1 = ResidualBlock([self.get_weights() , self.get_weights()],
                                 [self.get_bn_weights() , self.get_bn_weights()])

        self.conv_b3 = ConvBlock(conv_weights = self.get_weights() , bn_weights = self.get_bn_weights(), kernel_size = 3 , downsample = True)
        self.rb2 = []
        for i in range(2):
            self.rb2.append(ResidualBlock([self.get_weights() , self.get_weights()] ,
                                          [self.get_bn_weights() , self.get_bn_weights()]))

        self.conv_b4 = ConvBlock(conv_weights = self.get_weights() , bn_weights = self.get_bn_weights(), kernel_size = 3 , downsample = True)
        self.rb3 = []
        for i in range(8):
            self.rb3.append(ResidualBlock([self.get_weights() , self.get_weights()] ,
                                          [self.get_bn_weights() , self.get_bn_weights()]))

        self.conv_b5 = ConvBlock(conv_weights = self.get_weights() , bn_weights = self.get_bn_weights(), kernel_size = 3 , downsample = True)
        self.rb4 = []
        for i in range(8):
            self.rb4.append(ResidualBlock([self.get_weights() , self.get_weights()] ,
                                          [self.get_bn_weights() , self.get_bn_weights()]))

        self.conv_b6 = ConvBlock(conv_weights = self.get_weights() , bn_weights = self.get_bn_weights(), kernel_size = 3 , downsample = True)
        self.rb5 = []
        for i in range(4):
            self.rb5.append(ResidualBlock([self.get_weights() , self.get_weights()] ,
                                          [self.get_bn_weights() , self.get_bn_weights()]))

        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()

    def load_pickle(self, file):
        with open(file , "rb") as content:
            return pickle.load(content)

    def get_weights(self):
        weights = self.darknet53_weights["conv_{}".format(self.conv_index)]
        self.conv_index += 1
        return weights

    def get_bn_weights(self):
        bn_weights = self.darknet53_bn["conv_{}".format(self.bn_index)]
        self.bn_index += 1
        return bn_weights

    def call(self, input , training = False):
        conv_b1 = self.conv_b1(input , training = training)
        conv_b2 = self.conv_b2(conv_b1 , training = training)
        residual_b1 = self.rb1(conv_b2 , training = training)

        conv_b3 = self.conv_b3(residual_b1 , training = training)
        residual_b2 = conv_b3
        for bi in range(len(self.rb2)):
            residual_b2 = self.rb2[bi](residual_b2 , training = training)

        conv_b4 = self.conv_b4(residual_b2, training = training)
        residual_b3 = conv_b4
        for bi in range(len(self.rb3)):
            residual_b3 = self.rb3[bi](residual_b3 , training = training)

        conv_b5 = self.conv_b5(residual_b3 , training = training)
        residual_b4 = conv_b5
        for bi in range(len(self.rb4)):
            residual_b4 = self.rb4[bi](residual_b4 , training = training)

        conv_b6 = self.conv_b6(residual_b4 , training = training)
        residual_b5 = conv_b6
        for bi in range(len(self.rb5)):
            residual_b5 = self.rb5[bi](residual_b5 , training = training)

        average_pool = self.avg_pool(residual_b5)
        output = tf.nn.bias_add(tf.matmul(tf.reshape(average_pool , [-1 , 1024]) , self.dense_weights["dense_kernel"]) , self.dense_weights["dense_bias"])
        return output
