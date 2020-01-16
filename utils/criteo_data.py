"""
crito dataset for ctr prediction
https://www.tensorflow.org/tutorials/structured_data/feature_columns#%E7%94%A8_tfdata_%E5%88%9B%E5%BB%BA%E8%BE%93%E5%85%A5%E6%B5%81%E6%B0%B4%E7%BA%BF
"""
import os
import sys
import math
import collections
import json
import multiprocessing as mlp
from functools import partial
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras



def text_line_wrap(columns):
    def textline_parse_csv(line):
        column_data = tf.io.decode_csv(line, [float()] * 14 + [str()] * 26)
        return (dict(zip(columns, column_data[1:])), column_data[0])
    return textline_parse_csv


def criteo_data_input(dataset_dir, feature_columns, epochs=5, sp_feat_dim=10, \
    batch_size=32, shuffle_buffer=1000000):
    """
    define the crito data input for factorization model
    """
    train_data_path = os.path.join(dataset_dir, "criteo.tra")
    dev_data_path = os.path.join(dataset_dir, "criteo.dev")
    vocab_path = os.path.join(dataset_dir, "spfeat.vocab")
    
    vocab_info = json.load(open(vocab_path, "r"))
    
    train_epoch_iters = math.ceil(vocab_info["train_count"] // batch_size)
    dev_epoch_iters = math.ceil(vocab_info["dev_count"] // batch_size)
    dense_features = ["I" + str(idx) for idx in range(1, 14)]
    sparse_features = ["C" + str(idx) for idx in range(1, 27)]
    
    tf_tra_dataset = tf.data.TextLineDataset(train_data_path).skip(1)
    tf_tra_dataset  = tf_tra_dataset.map(text_line_wrap(dense_features + sparse_features), \
        num_parallel_calls=16)
    tf_tra_dataset = tf_tra_dataset.shuffle(shuffle_buffer)
    tf_tra_dataset = tf_tra_dataset.repeat(epochs)
    tf_tra_dataset = tf_tra_dataset.batch(batch_size)
    tf_tra_dataset = tf_tra_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    tf_dev_dataset = tf.data.TextLineDataset(dev_data_path).skip(1)
    tf_dev_dataset  = tf_dev_dataset.map(text_line_wrap(dense_features + sparse_features), \
        num_parallel_calls=16)
    tf_dev_dataset = tf_dev_dataset.batch(batch_size)
    tf_dev_dataset = tf_dev_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return tf_tra_dataset, tf_dev_dataset, sparse_features, \
        dense_features, train_epoch_iters, dev_epoch_iters


def process_func(sparse_features, dense_features, part_data):
    results = []
    for row in part_data.iterrows():
        row = row[1]
        example = serialize_example(row, sparse_features, dense_features)
        results.append(example)
    return results


def make_tfrecords(dataset_path, tfrecord_path, chunksize=1000000):
    """
    """
    dense_features = ["I" + str(idx) for idx in range(1, 14)]
    sparse_features = ["C" + str(idx) for idx in range(1, 27)]
    datasets = pd.read_csv(dataset_path, chunksize=chunksize)
    count = 0
    partial_func = partial(process_func, sparse_features, dense_features)
    with tf.io.TFRecordWriter(tfrecord_path) as tra_writer:
        for chunk_data in datasets:
            pool = mlp.Pool(16)
            data_split = np.array_split(chunk_data, 16)
            results = pool.map(partial_func, data_split)
            pool.close()
            pool.join()

            for examples in results:
                for example in examples:
                    tra_writer.write(example)
            count += sum([len(s) for s in results])
            print("----have tacked count is---", count)


def serialize_example(row, sp_features, ds_features):
    """
    """
    feature = {}
    for sp_feat in sp_features:
        feature[sp_feat] = _bytes_feature(row[sp_feat].encode("utf-8"))
    for ds_feat in ds_features:
        feature[ds_feat] = _float_feature(row[ds_feat])
    feature["label"] = _float_feature(row["label"])
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_tfrecords(path, dataset_dir, feature_columns, phrase="train", \
    batch_size=32, epochs=5, sp_feat_dim=10, shuff_buffer=1000000):
    """
    """
    vocab_path = os.path.join(dataset_dir, "spfeat.vocab")
    vocab_info = json.load(open(vocab_path, "r"))

    dense_features = ["I" + str(idx) for idx in range(1, 14)]
    sparse_features = ["C" + str(idx) for idx in range(1, 27)]
    
    feature_columns_spec = feature_columns + [tf.feature_column.numeric_column("label")]
    
    def _parse_function(example_proto):
        """
        """
        parsed_example = tf.io.parse_single_example(example_proto, \
            features=tf.feature_column.make_parse_example_spec(feature_columns_spec))
        label = parsed_example.pop("label")
        label = tf.reshape(label, [-1])
        for feat in sparse_features:
            parsed_example[feat] = tf.sparse.to_dense(parsed_example[feat])
        return (parsed_example, label)

    train_dataset = tf.data.TFRecordDataset(path, num_parallel_reads=8)
    train_dataset = train_dataset.map(_parse_function, \
        num_parallel_calls=16)
    data_count = 0
    if phrase == "train":
        data_count = vocab_info["train_count"]
        train_dataset = train_dataset.shuffle(shuff_buffer)
        train_dataset = train_dataset.repeat(epochs)
    else:
        data_count = vocab_info["dev_count"]
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #for batch, label in train_dataset:
    #    print(batch)
    #    print(label)
    #    break
    epoch_iters = math.ceil(data_count // batch_size)
    return train_dataset, epoch_iters, sparse_features, dense_features


if __name__ == "__main__":
    dataset_dir = "/home/lyt/workspace/recsys/data/criteo_data/"
    dataset_path = "/home/lyt/workspace/recsys/data/criteo_data/train.txt"
    #data, _, _, _, _, _, _, = criteo_data_input(dataset_dir, sp_feat_dim=10, batch_size=256)
    #for batch, label in data.take(1):
    #    print(batch, label)
    #    break
    tra_dataset_path = "/home/lyt/workspace/recsys/data/criteo_data/criteo.tra"
    tfrecord_path = os.path.join(dataset_dir, "tra.tfrecords")
    #make_tfrecords(tra_dataset_path, tfrecord_path, chunksize=1000000)
    dev_dataset_path = "/home/lyt/workspace/recsys/data/criteo_data/criteo.dev"
    dev_tfrecord_path = os.path.join(dataset_dir, "dev.tfrecords")
    #make_tfrecords(dev_dataset_path, dev_tfrecord_path, chunksize=1000000)
    #read_tfrecords(tfrecord_path, dataset_dir, shuff_buffer=100)
