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
from run_exp import FM_model
from tf_impl_reco.CTR.inputs import *


class DeepFM(keras.layers.Layer):
    def __init__(self, fm_units, deep_units, field_count):
        super(DeepFM, self).__init__()
        self.fm_model = FM_model(fm_units, field_count)
        self.deep_layers = []
        self.batch_normalization_layers = []
        for idx, unit in enumerate(deep_units):
            self.deep_layers.append(keras.layers.Dense(unit, activation="relu"))
            self.batch_normalization_layers.append(tf.keras.layers.BatchNormalization())
        self.logit_dense = keras.layers.Dense(deep_units[-1])
        

    def call(self, dense_inputs, sparse_inputs):
        """
        """
        fm_output = self.fm_model(dense_inputs, sparse_inputs, True)
        deep_output = tf.concat([dense_inputs, sparse_inputs], axis=-1)
        for idx, layer in enumerate(self.deep_layers):
            deep_output = layer(deep_output)
            deep_output = self.batch_normalization_layers[idx](deep_output)
        deep_output = self.logit_dense(deep_output)
        deep_output = tf.reshape(deep_output, [-1])
        outputs = tf.math.sigmoid(fm_output + deep_output)
        outputs = tf.reshape(outputs, [-1])
        return outputs


def deepfm_model_def(sp_dim, deep_units, feature_columns, sp_feats, dense_feats):
    """
    @param fm_units
    @param deep_units
    @param feature_columns, tf feature columns list
    @param sp_feats, categorical feature name list
    @param dense_feats, dense feature name list
    """
    dense_inputs = {}
    sparse_inputs = {}
    inputs = {}
    # name param is necessary, for dict input
    for ds_feat in dense_feats:
        inputs[ds_feat] = keras.Input(shape=[1, ], name=ds_feat)
        dense_inputs[ds_feat] = inputs[ds_feat]
    for sp_feat in sp_feats:
        inputs[sp_feat] = keras.Input(shape=[1, ], dtype=tf.string, name=sp_feat)
        sparse_inputs[sp_feat] = inputs[sp_feat]

    dense_features = keras.layers.DenseFeatures(feature_columns[:len(dense_feats)])(dense_inputs)
    sparse_features = keras.layers.DenseFeatures(feature_columns[len(dense_feats):])(sparse_inputs)

    outputs = DeepFM(sp_dim, deep_units, len(sp_feats))(dense_features, sparse_features)
    model = keras.Model(inputs=inputs, outputs=outputs)

    loss = keras.losses.BinaryCrossentropy()
    metrics = [AUC_T()]
    model.compile("adam", loss=loss, metrics=metrics)

    return model


if __name__ == "__main__":
    import os
    import sys
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    sys.path.append("../../")
    from tf_impl_reco.utils.criteo_data import *

    #dataset_dir = "/home/lyt/workspace/recsys/data/criteo_data/debug_data/"
    dataset_dir = "/home/lyt/workspace/recsys/data/criteo_data/"
    epochs = 1
    shuff_buffer = 500000
    batch_size = 256
    sp_feat_dim = 10
    deep_units = [400, 400, 1]

    feature_columns = create_feature_columns(dataset_dir, sp_feat_dim, False)
    train_dataset, dev_dataset, sparse_features, dense_features, \
        train_epoch_iters, dev_epoch_iters = criteo_data_input(dataset_dir, feature_columns, \
            epochs=epochs, sp_feat_dim=sp_feat_dim, batch_size=batch_size, shuffle_buffer=shuff_buffer)
    
    fm_model = deepfm_model_def(sp_feat_dim, deep_units, feature_columns, sparse_features, dense_features)

    binary_loss = keras.losses.BinaryCrossentropy()
    for batch, label in train_dataset.take(1):
        assert(len(label.shape)==1)
        output = fm_model(batch)
        loss = binary_loss(label, output)
        print(loss)

    fm_model.fit(train_dataset, validation_data=dev_dataset, epochs=epochs, \
            steps_per_epoch=train_epoch_iters, validation_steps=dev_epoch_iters)
