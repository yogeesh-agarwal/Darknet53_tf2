import cv2
import numpy as np
import keras_datagenerator as pre_process

def test_preprocess():
    batch_size = 1
    num_batch = 1
    train_data_path = "/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_train/WIDER_train/images/"
    val_data_path = "/home/yogeesh/yogeesh/datasets/face_detection/wider face/WIDER_val/WIDER_val/images/"
    pos_train_file = "../data/wider_train_file.pickle"
    neg_train_file = "../data/neg_samples.pickle"
    val_file = "../data/wider_val_file.pickle"
    batch_generator = pre_process.DataGenerator(input_size = 416,
                                                batch_size = batch_size,
                                                data_path = train_data_path,
                                                pos_file = pos_train_file,
                                                neg_file = neg_train_file,
                                                is_norm = False,
                                                is_augment = True,
                                                )

    print(batch_generator.num_instances)
    for index in range(num_batch):
        batch_images , batch_labels = batch_generator.__getitem__(index)
        for i in range(len(batch_images)):
            image = batch_images[i]
            label = batch_labels[i]
            if label[0] == 1:
                string = "FACE"
            else:
                string = "NON FACE"
            print(string)
            cv2.imshow(string , image[:,:,::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    test_preprocess()
