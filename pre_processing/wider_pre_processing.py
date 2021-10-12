import numpy as np
import pickle
import os

def read_files(filepath):
    file = open(filepath , "r")
    lines = file.readlines()
    image_names = []
    count = 1
    while count < len(lines):
        valid_file = False
        length = int(lines[count].split("\n")[0])
        if lines[count] == "0\n":
            print("0 faces in the image")

        valid_file = True
        filename = lines[count-1][:-1]
        event_class = int(filename.split("--")[0])
        if length == 0:
            count += 3
            continue
        image_names.append(filename)
        count += 1
        count += (length + 1)
        if not valid_file:
            count += 1

    return image_names

if __name__ == "__main__":
    train_image_names = read_files("/home/yogeesh/yogeesh/datasets/face_detection/wider face/wider_face_split/wider_face_split/wider_face_train_bbx_gt.txt")
    val_image_names = read_files("/home/yogeesh/yogeesh/datasets/face_detection/wider face/wider_face_split/wider_face_split/wider_face_val_bbx_gt.txt")
    with open("./data/wider_train_file.pickle" , "wb") as f:
        pickle.dump(train_image_names , f , pickle.HIGHEST_PROTOCOL)
    print("training data is saved in pickle file")

    with open("./data/wider_val_file.pickle" , "wb") as f:
        pickle.dump(val_image_names , f , pickle.HIGHEST_PROTOCOL)
    print("validation data is saved in pickle file")
