"""
writen your own metrics

https://www.tensorflow.org/api_docs/python/tf/math/in_top_k?version=stable

https://www.tensorflow.org/api_docs/python/tf/math/top_k?version=stable
https://www.tensorflow.org/api_docs/python/tf/math/equal?version=stable
https://www.tensorflow.org/api_docs/python/tf/where?version=stable

https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric?version=stable

https://github.com/tensorflow/ranking/blob/cf0b066ff2cd7e195ba9dd6761bb2af622c42080/tensorflow_ranking/python/utils.py#L49
"""
import os
import sys
import math
import numpy as np
from scipy import stats
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


class DCG(keras.metrics.Metric):
    def __init__(self, k=10, name="ndgg", **kwargs):
        super(DCG, self).__init__(name=name, **kwargs)
        self.k = k
        # sum weight
        self.sum_weight = self.add_weight(name="ncg_sum", initializer="zeros")
        # total sample weight
        self.total_count = self.add_weight(name="sample_count", initializer="zeros")

    def update_state(self, y_true, y_pred):
        """
        @param: y_true, bs_size, seq_len
        @param: y_pred, bs_size, seq_len, num_units
        """
        for seq_idx in range(y_pred.shape[1]):
            y_true_idx = tf.expand_dims(y_true[:, seq_idx], axis=1)
            top_k_indices = tf.math.top_k(y_pred[:, seq_idx, :], k=self.k)[1]
            # find the position
            positions = tf.where(tf.math.equal(y_true_idx, top_k_indices))[:, 1]
            positions = tf.cast(positions, tf.float32)
            ncgs = tf.math.log(2.0) / tf.math.log(positions + 2)

            self.sum_weight.assign_add(tf.reduce_sum(ncgs))
            self.total_count.assign_add(y_pred.shape[0])
        
    def result(self):
        return tf.math.divide(self.sum_weight, self.total_count)

        
class MRR(keras.metrics.Metric):
    def __init__(self, k, name="mrr", **kwargs):
        super(MRR, self).__init__(name=name, **kwargs)
        self.k = k
        # sum weight
        self.sum_weight = self.add_weight(name="mrr_sum", initializer="zeros")
        # total sample weight
        self.total_count = self.add_weight(name="sample_count", initializer="zeros")
        
    def update_state(self, y_true, y_pred, mask):
        """
        @param y_true
        @param y_pred, had been masked
        @mask mask
        """
        mrr_vals = []
        for seq_idx in range(y_pred.shape[1]):
            y_true_idx = tf.expand_dims(y_true[:, seq_idx], axis=1)
            top_k_indices = tf.math.top_k(y_pred[:, seq_idx, :], k=self.k)[1]
            # find the position
            positions = tf.where(tf.math.equal(y_true_idx, top_k_indices))[:, 1]
            print("---", positions)
            # bs_size,
            positions = tf.cast(positions, tf.float32) + 1.0
            mrr_val = 1.0 / positions
            self.sum_weight.assign_add(tf.reduce_sum(mrr_vals))
            self.total_count.assign_add(tf.math.reduce_sum(positions))

    def result(self):
        return tf.math.divide(self.sum_weight, self.total_count)


def sequence_mrr(y_true, y_pred, mask):
    """
    numpy version
    @param y_true
    @param y_pred
    @param mask
    """
    y_true = y_true - 1
    #print("---", y_true.shape, mask.shape)
    mrr_vals = []
    for seq_idx in range(y_pred.shape[1]):
        pred_ranks = list(map(stats.rankdata, -y_pred[:, seq_idx, :]))
        label_rank = [val[idx] for val, idx in zip(pred_ranks, y_true[:, seq_idx])]
        mrr_val = [1.0 / rank for rank in label_rank]
        mrr_vals.append(np.array(mrr_val))
    mrr_vals = np.stack(mrr_vals, axis=1)
    mrr_vals = np.multiply(mrr_vals, mask)
    #print("---", mrr_vals.shape)
    return np.sum(mrr_vals), np.sum(mask)
    