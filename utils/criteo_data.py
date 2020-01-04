"""
crito dataset for ctr prediction
https://www.tensorflow.org/tutorials/structured_data/feature_columns#%E7%94%A8_tfdata_%E5%88%9B%E5%BB%BA%E8%BE%93%E5%85%A5%E6%B5%81%E6%B0%B4%E7%BA%BF
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras


def demo(example_batch, feature_column):
  feature_layer = keras.layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())



def prepare_input_data(dataset_path, test_size=0.1):
    """
    split train and dev dataset, get the data count
    make sp_features vocab
    """
    


def criteo_data_input(dataset_path, epochs=5, sp_feat_dim=10, batch_size=32):
    """
    define the crito data input for factorization model
    """
    dataset = pd.read_csv(dataset_path, sep="\t")
    dense_features = ["I" + str(idx) for idx in range(1, 14)]
    sparse_features = ["C" + str(idx) for idx in range(1, 27)]
    dataset.columns = ["label"] + dense_features + sparse_features
    dataset[dense_features] = dataset[dense_features].astype(np.float32)
    dataset[sparse_features] = dataset[sparse_features].astype(str)
    dataset[dense_features] = dataset[dense_features].fillna(0.0)
    dataset[sparse_features] = dataset[sparse_features].fillna('-1')
    
    train_data, dev_data = train_test_split(dataset, test_size=0.2)

    labels = train_data.pop("label")
    tf_tra_dataset = tf.data.Dataset.from_tensor_slices((dict(train_data), labels.values))
    tf_tra_dataset = tf_tra_dataset.shuffle(train_data.shape[0])
    tf_tra_dataset = tf_tra_dataset.repeat(epochs)
    tf_tra_dataset = tf_tra_dataset.batch(batch_size)

    labels = dev_data.pop("label")
    tf_dev_dataset = tf.data.Dataset.from_tensor_slices((dict(dev_data), labels.values))
    tf_dev_dataset = tf_dev_dataset.batch(batch_size)

    # for test
    #exaple_batch_data = next(iter(tf_dataset))[0]
    feature_columns = []
    for dense_feat in dense_features:
        feature_columns.append(tf.feature_column.numeric_column(dense_feat))
        #demo(exaple_batch_data, tf.feature_column.numeric_column(dense_feat))

    for sp_feat in sparse_features:
        categorical_column_feat = tf.feature_column.categorical_column_with_vocabulary_list(sp_feat, \
            dataset[sp_feat].unique())
        feature_columns.append(tf.feature_column.embedding_column(categorical_column_feat, sp_feat_dim))
        # demo(exaple_batch_data, tf.feature_column.indicator_column(categorical_column_feat))
        # demo(exaple_batch_data, tf.feature_column.embedding_column(categorical_column_feat, sp_feat_dim))

    return tf_tra_dataset, tf_dev_dataset, feature_columns, sparse_features, dense_features


if __name__ == "__main__":
    dataset_path = "/home/lyt/workspace/recsys/data/criteo_data/train.txt"
    criteo_data_input(dataset_path, sp_feat_dim=10, batch_size=2)
    pass
