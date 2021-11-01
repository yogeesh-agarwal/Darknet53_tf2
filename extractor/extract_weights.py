import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.training import py_checkpoint_reader

def get_weights(save_dir):
    latest_ckp = tf.train.latest_checkpoint(save_dir)
    reader = py_checkpoint_reader.NewCheckpointReader(latest_ckp)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_to_dtype_map = reader.get_variable_to_dtype_map()

    state_dict = {v:reader.get_tensor(v) for v in reader.get_variable_to_shape_map()}
    return state_dict

def gen_var_dict(state_dict):
    conv_layers = {}
    bn_layers = {}
    dense_layers = {}
    for key in state_dict:
        if "conv_layer" in key and "optimizer" not in key:
            conv_layers[key] = state_dict[key]
        elif "bn_layer" in key and "optimizer" not in key:
            bn_layers[key] = state_dict[key]
        elif "dense_layer" in key and "optimizer" not in key:
            dense_layers[key] = state_dict[key]

    return conv_layers , bn_layers , dense_layers

def get_layer_var_names(conv_index = -1 , sub_index = -1 , residual_blk_index = None):
    conv_layer_names = []
    bn_layer_names = []
    if residual_blk_index is not None:
        if sub_index == -1:
            conv_layer_names.append("residual_b1/residual_block1/conv_layer/kernel")
            conv_layer_names.append("residual_b1/residual_block2/conv_layer/kernel")
            bn_layer_names.append("residual_b1/residual_block1/bn_layer/")
            bn_layer_names.append("residual_b1/residual_block2/bn_layer/")
        else:
            conv_layer_names.append("residual_blocks_{}/{}/residual_block1/conv_layer/kernel".format(residual_blk_index , sub_index))
            conv_layer_names.append("residual_blocks_{}/{}/residual_block2/conv_layer/kernel".format(residual_blk_index , sub_index))
            bn_layer_names.append("residual_blocks_{}/{}/residual_block1/bn_layer/".format(residual_blk_index , sub_index))
            bn_layer_names.append("residual_blocks_{}/{}/residual_block2/bn_layer/".format(residual_blk_index , sub_index))

    elif conv_index >= 1:
        conv_layer_names.append("conv_b{}/conv_layer/kernel".format(conv_index))
        bn_layer_names.append("conv_b{}/bn_layer/".format(conv_index))

    features = ["gamma" , "beta" , "moving_mean" , "moving_variance"]
    for i in range(len(bn_layer_names)):
        var_names = []
        bn_layer_name = bn_layer_names[i]
        for feat in features:
            var_names.append("{}{}/.ATTRIBUTES/VARIABLE_VALUE".format(bn_layer_name , feat))
        bn_layer_names[i] = var_names

    for i in range(len(conv_layer_names)):
        conv_layer_name = conv_layer_names[i] + "/.ATTRIBUTES/VARIABLE_VALUE"
        conv_layer_names[i] = conv_layer_name

    return conv_layer_names , bn_layer_names

def get_layer_weights(conv_layer_names , bn_layer_names , conv_layers , bn_layers , conv_dict, bn_dict , layer_index):
    index = 0
    for conv_layer_name , bn_layer_name in zip(conv_layer_names , bn_layer_names):
        conv_dict["conv_{}".format(layer_index + index)] = conv_layers[conv_layer_name]
        bn_vars = {}
        features = ["gamma" , "beta" , "mean" , "variance"]
        for i in range(4):
            bn_vars[features[i]] = bn_layers[bn_layer_name[i]]
        bn_dict["conv_{}".format(layer_index + index)] = bn_vars
        index += 1

def get_block_weights(conv_dict , bn_dict, conv_layers , bn_layers , layer_index , residual_blk_index = 0 , count = 0 , conv_blk_index = 0):
    if conv_blk_index == -1 and residual_blk_index == -1:
        raise Exception("both conv_blk_index and residual_blk_index cannot be invalid")
    if conv_blk_index > 0 and residual_blk_index > 0:
        raise Exception("both indexes cannot be set at the same time.")
    if residual_blk_index > 1 and count < 0:
        raise Exception("please provide the number of repeat of this residaul block as count var.")

    if conv_blk_index > 0:
        conv_layer_names , bn_layer_names = get_layer_var_names(conv_index = conv_blk_index)
        get_layer_weights(conv_layer_names , bn_layer_names , conv_layers , bn_layers , conv_dict , bn_dict , layer_index)
    elif residual_blk_index > 0:
        if residual_blk_index == 1:
            assert count == 0
            conv_layer_names , bn_layer_names = get_layer_var_names(residual_blk_index = residual_blk_index)
            get_layer_weights(conv_layer_names , bn_layer_names , conv_layers , bn_layers , conv_dict , bn_dict , layer_index)
        else:
            for i in range(count):
                conv_layer_names , bn_layer_names = get_layer_var_names(residual_blk_index = residual_blk_index , sub_index = i)
                get_layer_weights(conv_layer_names , bn_layer_names , conv_layers , bn_layers , conv_dict , bn_dict , layer_index)
                layer_index += 2

def get_dense_weights(dense_dict , dense_layers):
    layer_name = "dense_layer/"
    features = ["kernel" , "bias"]
    for feat in features:
        dense_dict["dense_{}".format(feat)] = dense_layers[layer_name + feat + "/.ATTRIBUTES/VARIABLE_VALUE"]

def refactor_weights(conv_dict , bn_dict , conv_layers , bn_layers):
    layer_index = 0
    get_block_weights(conv_dict , bn_dict , conv_layers , bn_layers , layer_index , conv_blk_index = 1)
    layer_index += 1
    get_block_weights(conv_dict , bn_dict , conv_layers , bn_layers , layer_index , conv_blk_index = 2)
    layer_index += 1
    get_block_weights(conv_dict , bn_dict , conv_layers , bn_layers , layer_index , residual_blk_index = 1)
    layer_index += 2
    get_block_weights(conv_dict , bn_dict , conv_layers , bn_layers , layer_index , conv_blk_index = 3)
    layer_index += 1
    get_block_weights(conv_dict , bn_dict , conv_layers , bn_layers , layer_index , residual_blk_index = 2 , count = 2)
    layer_index += (2*2)
    get_block_weights(conv_dict , bn_dict , conv_layers , bn_layers , layer_index , conv_blk_index = 4)
    layer_index += 1
    get_block_weights(conv_dict , bn_dict , conv_layers , bn_layers , layer_index , residual_blk_index = 3 , count = 8)
    layer_index += (2*8)
    get_block_weights(conv_dict , bn_dict , conv_layers , bn_layers , layer_index , conv_blk_index = 5)
    layer_index += 1
    get_block_weights(conv_dict , bn_dict , conv_layers , bn_layers , layer_index , residual_blk_index = 4 , count = 8)
    layer_index += (2*8)
    get_block_weights(conv_dict , bn_dict , conv_layers , bn_layers , layer_index , conv_blk_index = 6)
    layer_index += 1
    get_block_weights(conv_dict , bn_dict , conv_layers , bn_layers , layer_index , residual_blk_index = 5 , count = 4)
    layer_index += (2*4)

def save_as_pickle(conv_dict , bn_dict , dense_dict , conv_file_path , bn_file_path , dense_file_path):
    with open(conv_file_path , "wb") as content:
        pickle.dump(conv_dict, content , protocol = pickle.HIGHEST_PROTOCOL)
    with open(bn_file_path , "wb") as content:
        pickle.dump(bn_dict , content , protocol = pickle.HIGHEST_PROTOCOL)
    with open(dense_file_path , "wb") as content:
        pickle.dump(dense_dict , content , protocol = pickle.HIGHEST_PROTOCOL)

    print("Darknet53 weights are extracted and saved in pickle files")

def extract_weights():
    state_dict = get_weights("../saved_models")
    conv_layers , bn_layers , dense_layers = gen_var_dict(state_dict)
    dense_dict = {}
    conv_dict = {}
    bn_dict = {}
    refactor_weights(conv_dict , bn_dict, conv_layers , bn_layers)
    get_dense_weights(dense_dict , dense_layers)
    save_as_pickle(conv_dict , bn_dict , dense_dict ,
                   "./extracted_weights/conv_weights.pickle" ,
                   "./extracted_weights/bn_weights.pickle" ,
                   "./extracted_weights/dense_weights.pickle")

if __name__ == "__main__":
    extract_weights()
