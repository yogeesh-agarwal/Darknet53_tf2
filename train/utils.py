import sys
import numpy as np
import tensorflow as tf

def class_loss(predictions , target):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.stop_gradient(target) , logits = predictions))
    tf.print("loss : " , loss , output_stream = "file://../data/losses.txt")
    return loss

def get_accuracy(prediction , ground_truths):
    tp = 0
    fp = 0
    for pred , gt in zip(prediction , ground_truths):
        index = np.argsort(pred)[::-1][0]
        if (gt[0] and index == 0) or (gt[1] and index == 1):
            tp += 1
        else:
            fp += 1
    accuracy = (tp / (tp + fp))
    return accuracy
