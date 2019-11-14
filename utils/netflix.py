"""
parse the netfloxprize dataset
"""
import sys
import os
import json
import numpy as np
import tensorflow as tf


def make_netflix_dataset(data_dir, train_lens, batch_size, epochs):
    """
    """
    tra_tfpath = os.path.join(data_dir, "tra.tfrecord")
    dev_tfpath = os.path.join(data_dir, "dev.tfrecord")
    test_tfpath = os.path.join(data_dir, "test.tfrecord")

    train_dataset = tf.data.TFRecordDataset(tra_tfpath)
    dev_dataset = tf.data.TFRecordDataset(dev_tfpath)
    test_dataset = tf.data.TFRecordDataset(test_tfpath)

    train_dataset = train_dataset.map(_parse_function)
    dev_dataset = dev_dataset.map(_parse_function)
    test_dataset = test_dataset.map(_parse_function)

    train_dataset = train_dataset.shuffle(train_lens)
    train_dataset = train_dataset.repeat(epochs)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(2)

    dev_dataset = dev_dataset.batch(batch_size)
    dev_dataset = dev_dataset.prefetch(2)

    test_dataset = dev_dataset.batch(batch_size)
    test_dataset = dev_dataset.prefetch(2)

    return train_dataset, dev_dataset, test_dataset


def make_tfrecords(train_dir, dev_path, test_path, data_dir):
    """
    """
    tra_c, dev_c, movie_vocab, user_vocab = \
        read_traindev_data(train_dir, dev_path, data_dir)
    test_c = read_test_data(test_path, data_dir, movie_vocab, user_vocab)
    return True, tra_c, dev_c, test_c

def read_traindev_data(train_dir, dev_path, data_dir):
    """
    get the dev ratings from train adn remove dev from train
    """
    dev_datas = {}
    with open(dev_path, "r") as reader:
        movie_id = ""
        for line in reader:
            if ":" in line:
                movie_id = line.strip().split(":")[0]
            else:
                userid = line.strip()
                dev_datas["%s_%s" %(movie_id, userid)] = 0
    
    user_id_map = {}
    movie_id_map = {}
    train_tfpath = os.path.join(data_dir, "tra.tfrecord")
    dev_tfpath = os.path.join(data_dir, "dev.tfrecord")
    train_count = 0
    dev_count = 0
    with tf.io.TFRecordWriter(train_tfpath) as tra_writer, \
        tf.io.TFRecordWriter(dev_tfpath) as dev_writer:
        for filename in os.listdir(train_dir):
            filepath = os.path.join(train_dir, filename)
            with open(filepath, "r") as reader:
                movie_id = next(reader).strip().split(":")[0]
                movie_id_map.setdefault(movie_id, len(movie_id))
                for user_line in reader:
                    userid, rating, timestamp = user_line.strip().split(",")
                    user_id_map.setdefault(userid, len(user_id_map))
                    sample = [movie_id_map[movie_id], user_id_map[userid], float(rating), timestamp]
                    example = serialize_example(sample[0], sample[1], sample[2])
                    if "%s_%s" %(movie_id, userid) not in dev_datas:
                        tra_writer.write(example)
                        train_count += 1
                    else:
                        dev_writer.write(example)
                        dev_count += 1
    print("-----train and dev dataset length----", train_count, dev_count)
    print("---vocab summary----", len(user_id_map), len(movie_id_map))
    json.dump({"user_vocab": user_id_map, "movie_vocab": user_id_map, \
        "train_count": train_count, "dev_count": dev_count}, \
        open(os.path.join(data_dir, "datas.dump"), "w"))

    return train_count, dev_count, movie_id_map, user_id_map


def read_test_data(test_path, data_dir, movie_vocab, user_vocab):
    """
    """
    test_datas_count = 0
    test_tfpath = os.path.join(data_dir, "test.tfrecord")
    with tf.io.TFRecordWriter(test_tfpath, "w") as test_tfwriter:
        with open(test_path, "r") as reader:
            movie_id = ""
            for line in reader:
                if ":" in line:
                    movie_id = line.strip().split(":")[0]
                else:
                    userid, timestamp = line.strip().split(",")
                    sample = [movie_vocab[movie_id], user_vocab[userid], 0, timestamp]
                    example = serialize_example(sample[0], sample[1], sample[2])
                    test_tfwriter.write(example)
                    test_datas_count += 1
    print("----test dataset cout------", test_datas_count)
    return test_datas_count


def serialize_example(movieid, userid, rating):
    """
    """
    feature = {
        "movie_id": _int_feature(movieid),
        "user_id": _int_feature(userid),
        "rating": _float_feature(rating)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _parse_function(example_proto):
    feature_description = {
        'movie_id': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'user_id': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'rating': tf.io.FixedLenFeature([], tf.float, default_value=0.0)
        }
    return tf.io.parse_single_example(example_proto, feature_description)


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == "__main__":
    train_dir = "/home/lyt/workspace/recsys/data/netflixprize/training_set"
    dev_path = "/home/lyt/workspace/recsys/data/netflixprize/probe.txt"
    test_path = "/home/lyt/workspace/recsys/data/netflixprize/qualifying.txt"
    data_dir = "/home/lyt/workspace/recsys/tf_impl_reco/data/"
    #read_traindev_data(train_dir, dev_path, data_dir)

    batch_size = 64
    epochs = 5
    make_netflix_dataset(train_dir, dev_path, test_path, data_dir, batch_size, epochs, cache=True)