"""
Factorization Machines(Steffen Rendle)
"""
import os
import sys
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("../../")
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tf_impl_reco.utils.metrics import AUC_T
from tf_impl_reco.utils.criteo_data import *
from tf_impl_reco.CTR.inputs import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.config.experimental.set_memory_growth(physical_devices[1], True)
from sklearn.metrics import roc_auc_score


class FM_model(keras.layers.Layer):
    def __init__(self, ds_field_count, sp_field_count, sp_dim, initializer, **kwargs):
        super(FM_model, self).__init__(kwargs)
        self.ds_field_count = ds_field_count
        self.sp_field_count = sp_field_count
        self.sp_dim = sp_dim
        #regularizer = tf.keras.regularizers.l2(1e-4)
        regularizer = None
        self.dense_embedding = keras.layers.Embedding(ds_field_count, sp_dim, \
            embeddings_regularizer=regularizer, embeddings_initializer=initializer)
        self.dense_linear = keras.layers.Dense(1, kernel_regularizer=regularizer, use_bias=False)
        self.bias = tf.Variable([0.0])
        self.batch_normalization = keras.layers.BatchNormalization()
    
    def fm_call_impl1(self, dense_inputs, sparse_inputs, logits=False):
        # first order part, (bs_size, 1)
        dense_linear_output = tf.reshape(self.dense_linear(dense_inputs), [-1])
        linear_logits = dense_linear_output + tf.math.reduce_sum(sparse_inputs, axis=-1)

        # second order part, name, from right to left, operate on field dimension
        # bs_size, field_count, factor_dim
        sparse_inputs = tf.reshape(sparse_inputs, [-1, self.sp_field_count, self.sp_dim])
        # bs_size, 1, k
        square_of_sum = tf.square(tf.math.reduce_sum(sparse_inputs, axis=1, keepdims=True))
        sum_of_square = tf.math.reduce_sum(sparse_inputs * sparse_inputs, axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        # last operation, reduce on the factor dimension
        cross_term = 0.5 * tf.math.reduce_sum(cross_term, axis=2, keepdims=False)
        output_logits = tf.reshape(cross_term, [-1]) + linear_logits
        if not logits:
            outputs = tf.math.sigmoid(output_logits)
        else:
            outputs = output_logits
        return outputs

    def fm_call_impl2(self, dense_inputs, sparse_inputs, logits=False):
        # construct the dense input, (bs_size, ds_field_count, dim)
        dense_inputs_vals = tf.expand_dims(dense_inputs, axis=2)
        dense_weights = tf.expand_dims(self.dense_embedding(tf.range(0, self.ds_field_count)), axis=0)
        dense_inputs_emb = tf.math.multiply(dense_weights, dense_inputs_vals)
        # concat the dense and sparse input
        sparse_inputs = tf.reshape(sparse_inputs, [-1, self.sp_field_count, self.sp_dim])
        inputs = tf.concat([dense_inputs_emb, sparse_inputs], axis=1)
        #inputs = self.batch_normalization(inputs)
        # first order
        linear_logits = tf.reduce_sum(inputs[:, :, 0], axis=1)
        linear_logits = tf.reshape(linear_logits, [-1])
        # second order part, name, from right to left, operate on field dimension
        # bs_size, field_count, factor_dim
        second_order_inputs = inputs[:, :, 1:]
        # bs_size, 1, k
        square_of_sum = tf.square(tf.math.reduce_sum(second_order_inputs, axis=1))
        sum_of_square = tf.math.reduce_sum(second_order_inputs * second_order_inputs, axis=1)
        cross_term = square_of_sum - sum_of_square
        # last operation, reduce on the factor dimension
        cross_term = 0.5 * tf.math.reduce_sum(cross_term, axis=-1, keepdims=False)
        output_logits = tf.reshape(cross_term, [-1]) + linear_logits + self.bias
        
        self.add_loss(1e-4 * tf.nn.l2_loss(inputs))

        if not logits:
            outputs = tf.math.sigmoid(output_logits)
        else:
            outputs = output_logits
        
        return outputs

    def call(self, dense_inputs, sparse_inputs, logits=False):
        """
        @param inputs,  batch_size, input_dim
        """
        return self.fm_call_impl2(dense_inputs, sparse_inputs, logits=False)


def fm_model_func(sp_feat_dim, feature_columns, sp_feats, dense_feats, initializer):
    """
    define the model
    bug need to understand, as the tutorial, label is (batch, ), then the output is should be the same not ,batch_size, 1
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
    outputs = FM_model(len(dense_feats), len(sparse_inputs), sp_feat_dim, initializer)(\
        dense_features, sparse_features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    loss = keras.losses.BinaryCrossentropy()
    metrics = [AUC_T()]
    optimizer = keras.optimizers.Adam(1e-4)
    model.compile(optimizer="adam", loss=loss, metrics=metrics)
    
    return model


if __name__ == "__main__":
    dataset_dir = "/home/lyt/workspace/recsys/data/criteo_data/"
    # for debug
    #dataset_dir = "/home/lyt/workspace/recsys/data/criteo_data/debug_data"
    epochs = 2
    shuff_buffer = 500000
    batch_size = 2048
    sp_feat_dim = 11
    initializer = tf.keras.initializers.RandomUniform(seed=1024)
    feature_columns = create_feature_columns(dataset_dir, sp_feat_dim, initializer, is_debug=False)
 
    train_dataset, dev_dataset, sparse_features, dense_features, \
        train_epoch_iters, dev_epoch_iters = criteo_data_input(dataset_dir, feature_columns, \
            epochs=epochs, sp_feat_dim=sp_feat_dim, batch_size=batch_size, shuffle_buffer=shuff_buffer)
    
    fm_model = fm_model_func(sp_feat_dim, feature_columns, sparse_features, dense_features, initializer)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./data/summary/fm/", histogram_freq=1)
    fm_model.fit(train_dataset, validation_data=dev_dataset, epochs=epochs, \
            steps_per_epoch=train_epoch_iters, validation_steps=dev_epoch_iters, callbacks=[tensorboard_callback])
