import os
import cv2
import pickle
import random
import numpy as np
import imgaug as ia
from operator import itemgetter
import imgaug.augmenters as iaa
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 input_size ,
                 batch_size ,
                 data_path ,
                 pos_file ,
                 neg_file ,
                 is_norm ,
                 is_augment ,
                 shuffle = False ,
                 instances = 2500):

        self.shuffle = shuffle
        self.is_norm = is_norm
        self.data_path = data_path
        self.is_augment = is_augment
        self.batch_size = batch_size
        self.input_width = input_size
        self.input_height = input_size
        self.num_instances = instances
        self.pos_data = self.load_pickle(pos_file)
        self.neg_data = self.load_pickle(neg_file)

        sometimes = lambda aug : iaa.Sometimes(0.4 , aug)
        self.augmentor = iaa.Sequential([
            iaa.Fliplr(0.3),
            sometimes(iaa.Affine(
                rotate = (-20 , 20),
                shear = (-13 , 13),
            )),
            iaa.SomeOf((0 , 5),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0 , 3.0)),
                    iaa.AverageBlur(k =  (2 , 7)),
                    iaa.MedianBlur(k = (3 , 11)),
                ]),
                iaa.Sharpen(alpha = (0 , 1.0) , lightness = (0.75 , 1.5)),
                iaa.AdditiveGaussianNoise(loc = 0 , scale = (0.0 , 0.05*255) , per_channel = 0.5),
                iaa.Add((-10 , 10) , per_channel = 0.5),
                iaa.Multiply((0.5 , 1.5) , per_channel = 0.5),
                iaa.LinearContrast((0.5 , 2.0) , per_channel = 0.5),
            ] , random_order = True)
        ] , random_order = True)

    def load_pickle(self , filepath):
        with open(filepath , "rb") as content:
            return pickle.load(content)

    def augment_data(self , images):
        aug_images = self.augmentor(images = images)
        return aug_images

    def encode_data(self , starting_index , ending_index):
        images = []
        labels = []
        for index in range(starting_index , ending_index):
            image_path = os.path.join(self.data_path , self.pos_data[index])
            image = cv2.cvtColor(cv2.imread(image_path) , cv2.COLOR_BGR2RGB)
            image = cv2.resize(image , (self.input_width , self.input_height))
            label = [1 , 0]
            images.append(image)
            labels.append(label)

        for index in range(starting_index , ending_index):
            try:
                neg_image_path = self.neg_data[index]
                image = cv2.cvtColor(cv2.imread(neg_image_path) , cv2.COLOR_BGR2RGB)
                image = cv2.resize(image , (self.input_width , self.input_height))
                label = [0 , 1]
                images.append(image)
                labels.append(label)
            except:
                print(index)

        augmented_images = images
        if self.is_augment:
            augmented_images = self.augment_data(images)

        encoded_images = []
        if self.is_norm:
            for image in augmented_images:
                norm_image = image / 255.0
                encoded_images.append(norm_image)

        else:
            encoded_images = augmented_images

        return encoded_images , labels

    def __getitem__(self, index):
        starting_index = index *  self.batch_size
        ending_index = starting_index + self.batch_size
        if ending_index > self.num_instances:
            ending_index = self.num_instances
            starting_index = ending_index - self.batch_size

        index_list = list(range(self.batch_size * 2))
        random.shuffle(index_list)
        encoded_images , encoded_labels = self.encode_data(starting_index , ending_index)
        encoded_images = list(itemgetter(*index_list)(encoded_images))
        encoded_labels = list(itemgetter(*index_list)(encoded_labels))

        image_data = np.array(encoded_images).reshape(self.batch_size * 2, self.input_height , self.input_width , 3)
        label_data = np.array(encoded_labels).reshape(self.batch_size * 2 , 2).astype(np.float32)
        return (image_data , label_data)

    def __len__(self):
        return int(np.ceil(self.num_instances / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.pos_data)
            np.random.shuffle(self.neg_data)
