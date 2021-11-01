import pickle
import numpy as np
import tensorflow as tf
from Darknet53_tf2.models.keras_model import Darknet53

def load_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def get_org_model():
    classifier = Darknet53(2)
    latest_chkpnt = tf.train.latest_checkpoint("../saved_models")
    classifier.load_weights(latest_chkpnt)
    return classifier

def get_all_layer_weights(model):
    weights = {}
    for layer in model.layers:
        weights[layer.name] = layer.get_weights()
    return weights

def seperate_conv_bn_weights(weights):
    conv_weights = {}
    bn_weights = {}
    dense_weights = {}
    index = 0
    for layer_name in weights:
        layer_weights = weights[layer_name]
        if "global_average_pooling2d" in layer_name:
            continue
        if "dense" in layer_name:
            dense_weights["kernel"] = layer_weights[0]
            dense_weights["bias"] = layer_weights[1]
            continue
        if "residual" in layer_name:
            for i in range(2):
                base_index = i*3
                conv_weight = layer_weights[base_index]
                bn_weight = {"gamma" : layer_weights[base_index+1],
                             "beta" : layer_weights[base_index+2]}
                if i == 0:
                    bn_weight["mean"] = layer_weights[6]
                    bn_weight["variance"] = layer_weights[7]
                else:
                    bn_weight["mean"] = layer_weights[8]
                    bn_weight["variance"] = layer_weights[9]
                conv_weights["conv_{}".format(index)] = conv_weight
                bn_weights["conv_{}".format(index)] = bn_weight
                index += 1
        else:
            bn_weight = {}
            conv_weight = layer_weights[0]
            bn_w = [layer_weights[i] for i in range(1,5)]
            bn_feat = ["gamma" , "beta" , "mean" , "variance"]
            for i in range(4):
                bn_weight[bn_feat[i]] = bn_w[i]
            conv_weights["conv_{}".format(index)] = conv_weight
            bn_weights["conv_{}".format(index)] = bn_weight
            index += 1

    return conv_weights, bn_weights , dense_weights

def compare_conv_weights(ext_weights , loaded_weights):
    for layer_name in ext_weights:
        layer_weight_ext = ext_weights[layer_name]
        layer_weight_loaded = loaded_weights[layer_name]
        print(layer_name , " : " , np.sum(np.abs(layer_weight_ext.flatten() - layer_weight_loaded.flatten())))

def compare_bn_weights(ext_weights , loaded_weights):
    features = ["gamma" , "beta" , "mean" , "variance"]
    for layer_name in ext_weights:
        layer_weight_ext = ext_weights[layer_name]
        layer_weight_loaded = loaded_weights[layer_name]
        print("*********** " , layer_name , " ****************")
        for feat in features:
            print(feat , " : " , np.sum(np.abs(layer_weight_ext[feat].flatten() - layer_weight_loaded[feat].flatten())))

def compare_bn_calls(input , bn_vars):
    gamma = bn_vars["gamma"]
    beta = bn_vars["beta"]
    mean = bn_vars["mean"]
    variance = bn_vars["variance"]
    bn_values = [bn_vars[key] for key in bn_vars]
    keras_bn = tf.keras.layers.BatchNormalization()
    keras_bn.build(input.shape)
    keras_bn.set_weights(bn_values)
    keras_out = keras_bn(input , training = False)

    tf_bn = tf.nn.batch_normalization(input , mean , variance , beta , gamma , variance_epsilon = 0.001)
    print(np.sum(np.abs(keras_out.numpy() - tf_bn.numpy())))

if __name__ == "__main__":
    ext_conv_weights = load_pickle("../extractor/extracted_weights/conv_weights.pickle")
    ext_bn_weights = load_pickle("../extractor/extracted_weights/bn_weights.pickle")
    ext_dense_weights = load_pickle("../extractor/extracted_weights/dense_weights.pickle")
    # classifier = get_org_model()
    # outs = classifier(np.random.uniform(size = (1,416,416,3)))
    # weights = get_all_layer_weights(classifier)
    # conv_weights , bn_weights , dense_weights = seperate_conv_bn_weights(weights)

    # compare_conv_weights(ext_conv_weights , conv_weights)
    # print("*****************************************\n"*2)
    # compare_bn_weights(ext_bn_weights , bn_weights)

    bn_input = np.random.uniform(size = (1,128,128,32))
    compare_bn_calls(bn_input , ext_bn_weights["conv_0"])
