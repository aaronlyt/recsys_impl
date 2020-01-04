"""
Factorization Machines(Steffen Rendle)
"""
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append("../../")
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tf_impl_reco.utils.metrics import AUC_T
from tf_impl_reco.utils.criteo_data import *


class FM_model(keras.layers.Layer):
    def __init__(self, hidden_dim, **kwargs):
        super(FM_model, self).__init__(kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        """
        """
        # input_dim, hidden_dim
        self.fm_vweights = self.add_weight(name="fm_v", shape=[self.hidden_dim, input_shape[-1]], \
            dtype=tf.float32)

    def call(self, inputs, logits=False):
        """
        @param inputs,  batch_size, input_dim
        """
        weights = tf.expand_dims(self.fm_vweights, axis=0)
        inputs_1 = tf.expand_dims(inputs, axis=2)
        sum_square = tf.math.reduce_sum(\
            tf.math.square(\
                tf.matmul(weights, inputs_1)
                ), axis=1)
        square_sum = tf.math.reduce_sum(\
            tf.matmul(\
                tf.square(weights), tf.square(inputs_1)
                ), axis=1)
        if not logits:
            outputs = tf.math.sigmoid(sum_square - square_sum)
        else:
            outputs = sum_square - square_sum
        outputs = tf.reshape(outputs, [-1])
        return outputs


def fm_model_func(hidden_dim, feature_columns, sp_feats, dense_feats):
    """
    define the model
    bug need to understand, as the tutorial, label is (batch, ), then the output is should be the same not ,batch_size, 1
    """
    inputs = {}
    # name param is necessary, for dict input
    for sp_feat in sp_feats:
        inputs[sp_feat] = keras.Input(shape=[1, ], dtype=tf.string, name=sp_feat)
    for ds_feat in dense_feats:
        inputs[ds_feat] = keras.Input(shape=[1, ], name=ds_feat)

    features = keras.layers.DenseFeatures(feature_columns)(inputs)
    outputs = FM_model(hidden_dim)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    loss = keras.losses.BinaryCrossentropy()
    metrics = [AUC_T()]
    model.compile(optimizer="adam", loss=loss, metrics=metrics)
    return model


if __name__ == "__main__":
    dataset_path = "/home/lyt/workspace/recsys/data/criteo_data/train_sample.txt"
    train_dataset, dev_dataset, feature_columns, sparse_features, dense_features = \
            criteo_data_input(dataset_path, batch_size=256)
    for batch, label in train_dataset.take(1):
        #print("---batch---", batch)
        print("---label---", label.shape)
    hidden_dim = 32
    sp_feat_dim = 10
    fm_model = fm_model_func(hidden_dim, feature_columns, sparse_features, dense_features)
    fm_model.fit(train_dataset, validation_data=dev_dataset)
