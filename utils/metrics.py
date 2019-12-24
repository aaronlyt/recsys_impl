"""
writen your own metrics

https://www.tensorflow.org/api_docs/python/tf/math/in_top_k?version=stable

https://www.tensorflow.org/api_docs/python/tf/math/top_k?version=stable
https://www.tensorflow.org/api_docs/python/tf/math/equal?version=stable
https://www.tensorflow.org/api_docs/python/tf/where?version=stable

https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric?version=stable
"""
import os
import sys
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def getHitRatio(y_true, y_pred, k=10):
    """
    """
    hits = [0] * len(y_true)
    for idx, pred in enumerate(y_pred):
        # true start from 1, and pred start from zeros
        top_k_p = np.argsort(pred, axis=-1)[-k:][::-1] + 1
        if y_true[idx] in top_k_p:
            hits[idx] = 1
    return hits

def getNCG(y_true, y_pred, k=10):
    ndcgs = []
    for idx, pred in enumerate(y_pred):
        top_k_p = np.argsort(pred, axis=-1)[-k:][::-1] + 1
        for pos, pred_item in enumerate(top_k_p):
            if pred_item == y_true[idx]:
                ndcgs.append(math.log(2) / math.log(pos + 2))
                break
    return ndcgs


class NDGG(keras.metrics.Metric):
    def __init__(self, k=10, name="ndgg", **kwargs):
        super(NDGG, self).__init__(name=name, **kwargs)
        # sum weight
        self.sum_weight = self.add_weight(name="ncg_sum", initializer="zeros")
        # total sample weight
        self.total_count = self.add_weight(name="sample_count", initializer="zeros")

    def update_state(self, y_true, y_pred):
        """
        """
        top_k_indices = tf.math.top_k(y_pred)[1]
        # find the position
        y_true = tf.expand_dims(y_true, axis=1)
        positions = tf.where(tf.math.equal(y_true, top_k_indices))[:, 1]
        positions = tf.cast(positions, tf.float32)
        ncgs = tf.math.log(2.0) / tf.math.log(positions + 2)

        self.sum_weight.assign_add(tf.reduce_sum(ncgs))
        self.total_count.assign_add(y_pred.shape[0])
    
    def result(self):
        return tf.math.divide(self.sum_weight, self.total_count)


class TopKPrecision(keras.metrics.Metric):
    def __init__(self, k=10, name="top_k_precision", **kwargs):
        super(NDGG, self).__init__(name=name, **kwargs)
        # sum weight
        self.sum_weight = self.add_weight(name="ncg_sum", initializer="zeros")
        # total sample weight
        self.total_count = self.add_weight(name="sample_count", initializer="zeros")

    def update_state(self, y_true, y_pred):
        """
        """
        top_k_indices = tf.math.top_k(y_pred)[1]
        # find the position
        y_true = tf.expand_dims(y_true, axis=1)
        positions = tf.where(tf.math.equal(y_true, top_k_indices))[:, 1]
        positions = tf.cast(positions, tf.float32)
        ncgs = tf.math.log(2.0) / tf.math.log(positions + 2)

        self.sum_weight.assign_add(tf.reduce_sum(ncgs))
        self.total_count.assign_add(y_pred.shape[0])
    
    def result(self):
        return tf.math.divide(self.sum_weight, self.total_count)