import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self , filter_shape , downsample = False , alpha = 0.1 , kernel_size = None):
        super().__init__()
        def get_pad_info(kernel_size):
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            return pad_beg , pad_end

        self.pad_beg , self.pad_end = 0 , 0
        self.strides = 1
        self.padding = "SAME"
        self.downsample = downsample
        if downsample:
            self.strides = 2
            self.padding = "VALID"
            if not kernel_size:
                raise("require an additional argument --> kernel_size")
            self.pad_beg , self.pad_end = get_pad_info(kernel_size)

        self.padded_input = tf.keras.layers.ZeroPadding2D(((self.pad_beg , self.pad_end) , (self.pad_beg , self.pad_end)))
        self.conv_layer = tf.keras.layers.Conv2D(filters = filter_shape[-1],
                                                 kernel_size = filter_shape[0],
                                                 strides = self.strides,
                                                 padding = self.padding,
                                                 use_bias = False,
                                                 kernel_regularizer = tf.keras.regularizers.l2(0.0005),
                                                 kernel_initializer = tf.random_normal_initializer(stddev = 0.01),
                                                 bias_initializer = tf.constant_initializer(0.))
        self.bn_layer = tf.keras.layers.BatchNormalization()
        self.leaky_Relu = tf.keras.layers.LeakyReLU(alpha = alpha)

    def call(self , input , training = False):
        if self.downsample:
            input = self.padded_input(input)
        conv_out = self.conv_layer(input)
        bn_out = self.bn_layer(conv_out , training = training)
        lr_out = self.leaky_Relu(bn_out)
        return lr_out

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self , filter_shapes):
        super().__init__()
        self.residual_block1 = ConvBlock(filter_shapes[0])
        self.residual_block2 = ConvBlock(filter_shapes[1])
        self.block_output = tf.keras.layers.Add()

    def call(self , input , training = False):
        shortcut = input
        residual_b1 = self.residual_block1(input , training = training)
        residual_b2 = self.residual_block2(residual_b1 , training = training)
        block_output = self.block_output([residual_b2 , shortcut])
        return block_output
