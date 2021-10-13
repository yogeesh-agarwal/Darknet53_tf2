import os
import sys
import pdb
import utils
import numpy as np
import tensorflow as tf
from Darknet53_tf2.models.keras_model import Darknet53
from Darknet53_tf2.pre_processing.keras_datagenerator import DataGenerator

def gen_callbacks(tb_ld , cp_path):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = tb_ld,
                                                          histogram_freq = 1,
                                                          update_freq = "epoch")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = cp_path,
                                                             verbose = 1,
                                                             mode = "max",
                                                             save_best_only = True,
                                                             monitor = "val_accuracy",
                                                             save_weights_only = True)
    return [tensorboard_callback , checkpoint_callback]

def get_epoch_number(latest_chkpnt):
    chkpnt_name = latest_chkpnt.split(".ckpt")[0]
    epoch_number = chkpnt_name.split("-")[1]
    return int(epoch_number)

def train(input_size ,
          batch_size,
          num_epochs,
          data_path ,
          val_data_path,
          pos_train_file,
          neg_train_file,
          val_file,
          neg_val_file,
          is_norm,
          is_augment,
          num_classes,
          save_dir,
          logs_dir):

    train_summary_writer = tf.summary.create_file_writer(logs_dir)
    tf.summary.experimental.set_step(0)
    callbacks = gen_callbacks(logs_dir+"/loss" , save_dir+"cp-{epoch:04d}.ckpt")
    train_data_generator = DataGenerator(input_size ,
                                   batch_size,
                                   data_path ,
                                   pos_train_file ,
                                   neg_train_file ,
                                   is_norm ,
                                   is_augment ,
                                   instances = 2500,
                                   shuffle = True)
    val_data_generator = DataGenerator(input_size ,
                                   batch_size,
                                   val_data_path ,
                                   val_file ,
                                   neg_val_file ,
                                   is_norm ,
                                   is_augment ,
                                   instances = 250,
                                   shuffle = False)

    classifier = Darknet53(num_classes , train_summary_writer)
    epoch_number = 0
    if not os.listdir(save_dir):
        print("No trained model found , starting from scratch")
    else:
        latest_chkpnt = tf.train.latest_checkpoint(save_dir)
        classifier.load_weights(latest_chkpnt)
        epoch_number = get_epoch_number(latest_chkpnt)
        print("partially trained model found , resuming the training for epoch : " , epoch_number)

    classifier.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3))
    classifier.fit(train_data_generator,
                   epochs = num_epochs,
                   use_multiprocessing = True,
                   validation_data = val_data_generator,
                   initial_epoch = epoch_number,
                   callbacks = callbacks)
    print("Training Completed")

def main():
    input_size = 256
    batch_size = 8
    num_epochs = 200
    train_data_path = "/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_train/WIDER_train/images/"
    val_data_path = "/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_val/WIDER_val/images/"
    pos_train_file = "../data/wider_train_file.pickle"
    neg_train_file = "../data/neg_samples.pickle"
    val_file = "../data/wider_val_file.pickle"
    neg_val_file = "../data/neg_samples_test.pickle"
    is_norm = True
    is_augment = True
    num_classes = 2
    save_dir = "../saved_models/"
    logs_dir = "../logs/"

    train(input_size,
          batch_size,
          num_epochs,
          train_data_path,
          val_data_path,
          pos_train_file,
          neg_train_file,
          val_file,
          neg_val_file,
          is_norm,
          is_augment,
          num_classes,
          save_dir,
          logs_dir)

if __name__ == "__main__":
    main()
