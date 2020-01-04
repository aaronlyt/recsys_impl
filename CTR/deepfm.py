"""
deepfm model
"""

import os
import sys
sys.path.append("../../")
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tf_impl_reco.utils.metrics import AUC_T
from fm import FM_model


class DeepFM(keras.layers.Layer):
    def __init__(self, fm_units, deep_units):
        super(DeepFM, self).__init__()
        self.fm_model = FM_model(fm_units)
        self.deep_layers = []
        for idx, unit in enumerate(deep_units):
            if idx != len(deep_units) - 1:
                self.deep_layers.append(keras.layers.Dense(unit, activation="relu"))
            else:
                self.deep_layers.append(keras.layers.Dense(unit))
    def call(self, inputs):
        """
        """
        fm_output = self.fm_model(inputs, True)
        deep_output = inputs
        for layer in self.deep_layers:
            deep_output = layer(deep_output)
        outputs = tf.reshape(deep_output, [-1])
        outputs = tf.math.sigmoid(fm_output + deep_output)
        return outputs


def deepfm_model_def(fm_units, deep_units, feature_columns, sp_feats, dense_feats):
    """
    @param fm_units
    @param deep_units
    @param feature_columns, tf feature columns list
    @param sp_feats, categorical feature name list
    @param dense_feats, dense feature name list
    """
    inputs = {}
    for sp_feat in sp_feats:
        inputs[sp_feat] = keras.Input([1,], dtype=tf.string, name=sp_feat)
    for ds_feat in dense_feats:
        inputs[ds_feat] = keras.Input([1,], dtype=tf.float32, name=ds_feat)

    ds_feats_layer = keras.layers.DenseFeatures(feature_columns)(inputs)
    outputs = DeepFM(fm_units, deep_units)(ds_feats_layer)
    model = keras.Model(inputs=inputs, outputs=outputs)

    loss = keras.losses.BinaryCrossentropy()
    metrics = [AUC_T()]
    model.compile("adam", loss=loss, metrics=metrics)

    return model


if __name__ == "__main__":
    import os
    import sys
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.append("../../")
    from tf_impl_reco.utils.criteo_data import *

    inputs = np.random.sample([2, 10])
    inputs = tf.constant(inputs, dtype=tf.float32)
    fm_model = DeepFM(20, [20, 30, 1])
    print(fm_model(inputs))
    
    dataset_path = "/home/lyt/workspace/recsys/data/criteo_sample.txt"
    train_dataset, dev_dataset, feature_columns, sparse_features, dense_features= criteo_data_input(dataset_path)
    for batch, label in train_dataset.take(-1):
        #print("---batch---", batch)
        assert(len(label.shape)==1)
    deep_units = [400, 400, 1]
    fm_dim = 10
    sp_feat_dim = 10
    fm_model = deepfm_model_def(fm_dim, deep_units, feature_columns, sparse_features, dense_features)
    fm_model.fit(train_dataset, validation_data=dev_dataset)
    
    loss = keras.losses.BinaryCrossentropy()
    metrics = keras.metrics.AUC()
    #for batch, label in train_dataset.take(1):
        #outputs = fm_model(batch)
        #label = tf.expand_dims(label, axis=1)
        #print("loss", loss(label, outputs))