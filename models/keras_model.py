from Darknet53_tf2.models.layers import ConvBlock , ResidualBlock
from Darknet53_tf2.train import utils
import tensorflow as tf
import sys

class Darknet53(tf.keras.Model):
    def __init__(self , num_classes , summary_writer = None):
        super(Darknet53, self).__init__()
        self.conv_b1 = ConvBlock([3,3,3,32])
        self.conv_b2 = ConvBlock([3,3,32,64] , downsample = True , kernel_size = 3)
        self.residual_b1 = ResidualBlock([[1,1,64,32] , [3,3,32,64]])

        self.conv_b3 = ConvBlock([3,3,64,128] , downsample = True , kernel_size = 3)
        self.residual_blocks_2 = []
        for _ in range(2):
            self.residual_blocks_2.append(ResidualBlock([[1,1,128,64] , [3,3,64,128]]))

        self.conv_b4 = ConvBlock([3,3,128,256] , downsample = True , kernel_size = 3)
        self.residual_blocks_3 = []
        for _ in range(8):
            self.residual_blocks_3.append(ResidualBlock([[1,1,256,128] , [3,3,128,256]]))

        self.conv_b5 = ConvBlock([3,3,256,512] , downsample = True , kernel_size = 3)
        self.residual_blocks_4 = []
        for _ in range(8):
            self.residual_blocks_4.append(ResidualBlock([[1,1,512,256] , [3,3,256,512]]))

        self.conv_b6 = ConvBlock([3,3,512,1024] , downsample = True , kernel_size = 3)
        self.residual_blocks_5 = []
        for _ in range(4):
            self.residual_blocks_5.append(ResidualBlock([[1,1,1024,512] , [3,3,512,1024]]))

        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense_layer = tf.keras.layers.Dense(units = num_classes,
                                                 use_bias = True)

        # initalize own metrics to keep track of loss and mean_abs_error
        self.loss_tracker = tf.keras.metrics.Mean(name = "loss")
        self.val_acc = tf.keras.metrics.CategoricalAccuracy(name = "val_accuracy")
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name = "mae")
        # define summary writer
        self.summary_writer = summary_writer

    def call(self , input , training = False):
        conv_b1 = self.conv_b1(input , training = training)
        conv_b2 = self.conv_b2(conv_b1 , training = training)
        residual_b1 = self.residual_b1(conv_b2 , training = training)

        conv_b3 = self.conv_b3(residual_b1 , training = training)
        residual_b2 = conv_b3
        for bi in range(len(self.residual_blocks_2)):
            residual_b2 = self.residual_blocks_2[bi](residual_b2 , training = training)

        conv_b4 = self.conv_b4(residual_b2, training = training)
        residual_b3 = conv_b4
        for bi in range(len(self.residual_blocks_3)):
            residual_b3 = self.residual_blocks_3[bi](residual_b3 , training = training)

        conv_b5 = self.conv_b5(residual_b3 , training = training)
        residual_b4 = conv_b5
        for bi in range(len(self.residual_blocks_4)):
            residual_b4 = self.residual_blocks_4[bi](residual_b4 , training = training)

        conv_b6 = self.conv_b6(residual_b4 , training = training)
        residual_b5 = conv_b6
        for bi in range(len(self.residual_blocks_5)):
            residual_b5 = self.residual_blocks_5[bi](residual_b5 , training = training)

        average_pool = self.avg_pool(residual_b5)
        output = self.dense_layer(average_pool)
        return output

    def train_step(self, data):
        # overide this function to govern what happens exactly in 'fit' function (while training)
        images , ground_truths = data

        with tf.GradientTape() as tape:
            predictions = self(images , training = True)
            loss = utils.class_loss(predictions , ground_truths)
            if self.summary_writer is not None:
                with self.summary_writer.as_default():
                    tf.summary.scalar("loss" , loss)

        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss , trainable_variables)
        self.optimizer.apply_gradients(zip(gradients , trainable_variables))

        #compute metrics
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(ground_truths , predictions)
        return {"loss" : self.loss_tracker.result() , "mae" : self.mae_metric.result()}

    def test_step(self , data):
        # override this function to govern what happens exactly during "evaluate" fucntion.
        images , ground_truths = data
        predictions = self(images , training = False)
        loss = utils.class_loss(predictions , ground_truths)
        self.val_acc.update_state(ground_truths, predictions)
        return {"loss" : loss , "accuracy" : self.val_acc.result()}

    @property
    def metrics(self):
        # we need to reset_state() for each metric in starting of each epoch,
        # mentioning our own metrics here will automatically tigger keras to
        # reset state on these metrics.
        return [self.loss_tracker , self.mae_metric , self.val_acc]

def compile_standalone():
    model = Darknet53(2)
    model.build(input_shape = [1 , 256 , 256 , 3])
    model.call(tf.keras.layers.Input(shape = (256 , 256 , 3)))
    model.summary()
