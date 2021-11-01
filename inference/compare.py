import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
from Darknet53_tf2.train import utils
from Darknet53_tf2.models.keras_model import Darknet53
from Darknet53_tf2.inference.infer_model import Darknet53Infer
from Darknet53_tf2.pre_processing.keras_datagenerator import DataGenerator

def load_pickle(file):
    with open(file , "rb") as content:
        return pickle.load(content)

def get_test_data(data_path , test_data , starting_index , ending_index):
    count = 0
    test_images = []
    # np.random.shuffle(test_data)
    for index in range(starting_index , ending_index):
        if count < num_images:
            # test_data[index] = test_data[index].replace("../" , "./")
            img = cv2.imread(os.path.join(data_path , test_data[index]))
            if img is None:
                raise Exception(data_path  , test_data[index])
            test_images.append(img)
            count += 1
    return test_images

def pre_process(images):
    preprocessed_images = []
    for image in images:
        resized_image = cv2.resize(image , (256 , 256))
        resized_image = cv2.cvtColor(resized_image , cv2.COLOR_BGR2RGB)
        preprocessed_images.append(resized_image / 255.0)

    return np.array(preprocessed_images).reshape(len(images) , 256 , 256 , 3)

def gen_test_label(is_face , num_images):
    label = [1 , 0] if is_face else [0 , 1]
    labels = [label for i in range(num_images)]
    return labels

def get_predictions(classifier,
                    data_path ,
                    test_pic_file ,
                    neg_test_pic_file ,
                    max_images ,
                    num_batches ,
                    batch_size ,
                    show_image):
    test_accuracy = 0
    test_data = load_pickle(test_pic_file)
    neg_test_data = load_pickle(neg_test_pic_file)
    for batch_id in range(num_batches):
        starting_index = batch_id * batch_size
        ending_index = min(starting_index + batch_size , max_images)
        num_images = ending_index - starting_index
        face_test_images = get_test_data(data_path , test_data , starting_index , ending_index)
        face_test_labels = gen_test_label(True , len(face_test_images))
        neg_test_images = get_test_data("" , neg_test_data , starting_index , ending_index)
        neg_test_labels = gen_test_label(False , len(neg_test_images))
        test_images = face_test_images + neg_test_images
        test_labels = face_test_labels + neg_test_labels

        inference_input = pre_process(test_images)
        predictions = classifier(inference_input , training = False)
        test_accuracy += utils.get_accuracy(predictions , test_labels)
        if show_image:
            for index in range(predictions.shape[0]):
                pred_index = np.argsort(predictions[index])[::-1][0]
                pred_class = "FACE" if pred_index == 0 else "NON FACE"
                cv2.imshow(pred_class , cv2.resize(test_images[index] , (416 , 416)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    test_accuracy /= num_batches
    return test_accuracy

def get_infer_classifier(conv_weights_file , bn_weight_file , dense_weights_file):
    classifier = Darknet53Infer(conv_weights_file ,
                               bn_weight_file ,
                               dense_weights_file)
    classifier.trainable = False
    return classifier

def compare_outs(out_1 , out_2):
    if out_1.shape != out_2.shape:
        raise Exception("Shape msimatch between both the outputs" , out_1.shape , out_2.shape)

    return np.sum(np.abs(out_1.flatten() - out_2.flatten()))

def compare_models(model1 , model2):
    test_images = get_test_data(data_path , load_pickle(test_pic_file) , 0 , 1)
    test_labels = gen_test_label(True , len(test_images))
    preprocessed_input = pre_process(test_images)
    model1_out = model1(preprocessed_input , training = False)
    model2_out = model2(preprocessed_input , training = False)
    return compare_outs(model1_out.numpy() , model2_out.numpy())

if __name__ == "__main__":
    num_images = 1000
    batch_size = 10
    num_classes = 2
    show_img = False
    inference_model = True
    checkpnt_dir = "../saved_models/"
    num_batches = num_images // batch_size
    test_pic_file = "../data/celeba_test_file.pickle"
    neg_test_pic_file = "../data/neg_samples_test.pickle"
    data_path = "/home/yogeesh/yogeesh/datasets/face_detection/celeba/img_align_celeba"
    if inference_model:
        conv_weights_file = "../extractor/extracted_weights/conv_weights.pickle"
        bn_weights_file = "../extractor/extracted_weights/bn_weights.pickle"
        dense_weights_file = "../extractor/extracted_weights/dense_weights.pickle"
        infer_classifier = get_infer_classifier(conv_weights_file , bn_weights_file , dense_weights_file)

    normal_classifier = Darknet53(num_classes)
    latest_chkpnt = tf.train.latest_checkpoint(checkpnt_dir)
    normal_classifier.load_weights(latest_chkpnt)
    diff = compare_models(infer_classifier, normal_classifier)
    print(diff)