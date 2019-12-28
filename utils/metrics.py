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
        ncg_vals = []
        for seq_idx in range(y_pred.shape[1]):
            y_true_idx = y_true[:, seq_idx]
            # this method come from tf ranking
            top_k_indices = tf.math.top_k(y_pred[:, seq_idx, :], k=self.k)[1]
            ranks = tf.argsort(top_k_indices) + 1
            # get the label ranks
            label_indices = tf.stack([tf.range(tf.shape(y_true_idx)[0]), \
                y_true_idx], axis=1)
            label_ranks = tf.gather_nd(ranks, label_indices)
            # bs_size,
            label_ranks = tf.cast(label_ranks, tf.float32)
            ncgs = tf.math.log(2.0) / tf.math.log(label_ranks + 2)
            ncg_vals.append(ncgs)
        ncg_vals = tf.stack(ncg_vals, axis=1)
        ncg_vals = tf.math.multiply(ncg_vals, mask)
        self.sum_weight.assign_add(tf.reduce_sum(ncg_vals))
        self.total_count.assign_add(tf.math.reduce_sum(mask))
        
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
            y_true_idx = y_true[:, seq_idx]
            # this method come from tf ranking
            top_k_indices = tf.math.top_k(y_pred[:, seq_idx, :], k=self.k)[1]
            ranks = tf.argsort(top_k_indices) + 1
            # get the label ranks
            label_indices = tf.stack([tf.range(tf.shape(y_true_idx)[0]), \
                y_true_idx], axis=1)
            label_ranks = tf.gather_nd(ranks, label_indices)
            # bs_size,
            positions = tf.cast(label_ranks, tf.float32)
            mrr_val = 1.0 / positions
            mrr_vals.append(mrr_val)
        mrr_vals = tf.stack(mrr_vals, axis=1)
        mrr_vals = tf.math.multiply(mrr_vals, mask)
        self.sum_weight.assign_add(tf.reduce_sum(mrr_vals))
        self.total_count.assign_add(tf.math.reduce_sum(mask))

    def result(self):
        return tf.math.divide(self.sum_weight, self.total_count)


def sequence_mrr(y_true, y_pred, mask):
    """
    numpy version
    @param y_true
    @param y_pred
    @param mask
    """
    #y_true = y_true - 1
    #print("---", y_true.shape, mask.shape)
    mrr_vals = []
    for seq_idx in range(y_pred.shape[1]):
        pred_ranks = list(map(stats.rankdata, -y_pred[:, seq_idx, :]))
        label_rank = [val[idx] for val, idx in zip(pred_ranks, y_true[:, seq_idx])]
        mrr_val = [1.0 / rank for rank in label_rank]
        mrr_vals.append(np.array(mrr_val))
    mrr_vals = np.stack(mrr_vals, axis=1)
    mrr_vals = np.multiply(mrr_vals, mask)
    print("---", mrr_vals.shape)
    return np.sum(mrr_vals), np.sum(mask)


if __name__ == "__main__":
    label = tf.constant([[1, 2, 0], [2, 1, 1], \
        [1, 0, 0]], dtype=tf.int32)
    mask = tf.constant([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], \
        [1.0, 0.0, 0.0]], dtype=tf.float32)
    import numpy as np
    np.random.seed(seed=1000)
    preds = tf.constant(np.random.random_sample([3, 3, 3]))
    print(preds)
    print(sequence_mrr(label.numpy(), preds.numpy(), mask.numpy()))

    m = MRR(3)
    m.update_state(label, preds, mask)
    print(m.result())