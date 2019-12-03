"""
parse the netfloxprize dataset
"""
import sys
import os
import math
import time
import datetime
import json
import multiprocessing as mlp
from functools import partial
import numpy as np
import pandas as pd
import tensorflow as tf


def read_traindev_arraydata(data_dir, train_dir, dev_path):
    """
    get the dev ratings from train adn remove dev from train
    time consuming: about 25min
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
    train_count = 0
    dev_count = 0
    train_dataset = []
    dev_dataset = []
    
    for filename in os.listdir(train_dir):
        filepath = os.path.join(train_dir, filename)
        with open(filepath, "r") as reader:
            movie_id = next(reader).strip().split(":")[0]
            movie_id_map.setdefault(movie_id, len(movie_id_map))
            for user_line in reader:
                userid, rating, timestamp = user_line.strip().split(",")
                user_id_map.setdefault(userid, len(user_id_map))
                sample = [movie_id_map[movie_id], user_id_map[userid], float(rating), timestamp]
                if "%s_%s" %(movie_id, userid) not in dev_datas:
                    train_dataset.append(sample)
                    train_count += 1
                else:
                    dev_dataset.append(sample)
                    dev_count += 1
    print("-----train and dev dataset length----", train_count, dev_count)
    print("---vocab summary----", len(user_id_map), len(movie_id_map))
    
    json.dump({"user_vocab": user_id_map, "movie_vocab": movie_id_map, \
        "train_count": train_count, "dev_count": dev_count}, \
        open(os.path.join(data_dir, "datas_arr.dump"), "w"))
    
    train_path = os.path.join(data_dir, "train" + ".csv")
    columns = ["movieid", "userid", "rating", "timestamp"]
    dataframe = pd.DataFrame(train_dataset, columns=columns)
    dataframe["movieid"] = dataframe["movieid"].astype(np.int32)
    dataframe["userid"] = dataframe["userid"].astype(np.int32)
    dataframe["rating"] = dataframe["rating"].astype(np.float32)
    dataframe.sort_values(by=["userid"], inplace=True)
    dataframe.to_csv(train_path, index=False, chunksize=1e7)
    
    dev_path = os.path.join(data_dir, "dev" + ".csv")
    dev_dataframe = pd.DataFrame(dev_dataset, columns=columns)
    dev_dataframe.sort_values(by=["userid"], inplace=True)
    dev_dataframe.to_csv(dev_path, index=False)

    return dataframe, dev_dataframe


def make_netflix_tensor_dataset(data_dir, batch_size, epochs, shuffle_len):
    """
    """
    #columns = ["userid", "movieid", "item_time_bias", "user_time_dev", "rating"]
    dtype = {"userid": np.int32, "movieid": np.int32, "item_time_bias": np.int32, \
            "user_time_dev": np.float32, "rating": np.float32}
    train_path = os.path.join(data_dir, "train_processed" + ".csv")
    dev_path = os.path.join(data_dir, "dev_processed" + ".csv")

    train_dataframe = pd.read_csv(train_path, dtype=dtype)
    dev_dataframe = pd.read_csv(dev_path, dtype=dtype)
    train_len = len(train_dataframe)
    dev_len = len(dev_dataframe)

    dataset = []
    train_dataset = tf.data.Dataset.from_tensor_slices(\
        ({"user_id": train_dataframe["userid"].values, \
            "movie_id": train_dataframe["movieid"].values, \
                "item_time_bias":train_dataframe["item_time_bias"].values, \
                    "user_time_dev":train_dataframe["user_time_dev"].values}, \
                        train_dataframe["rating"].values))
    dev_dataset = tf.data.Dataset.from_tensor_slices(\
        ({"user_id": dev_dataframe["userid"].values, \
            "movie_id": dev_dataframe["movieid"].values, \
                "item_time_bias":dev_dataframe["item_time_bias"].values, \
                    "user_time_dev":dev_dataframe["user_time_dev"].values}, \
                        dev_dataframe["rating"]))
    
    train_dataset = train_dataset.shuffle(shuffle_len)
    train_dataset = train_dataset.repeat(epochs)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    dev_dataset = dev_dataset.batch(batch_size)

    return train_dataset, dev_dataset, train_len, dev_len


def make_netflix_dataset(data_dir, train_lens, batch_size, epochs, buffer_size):
    """
    """
    tra_tfpath = os.path.join(data_dir, "tra.tfrecord")
    dev_tfpath = os.path.join(data_dir, "dev.tfrecord")
    test_tfpath = os.path.join(data_dir, "test.tfrecord")

    train_dataset = tf.data.TFRecordDataset(tra_tfpath, num_parallel_reads=8)
    dev_dataset = tf.data.TFRecordDataset(dev_tfpath)
    test_dataset = tf.data.TFRecordDataset(test_tfpath)

    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.repeat(epochs)
    train_dataset = train_dataset.map(_parse_function, \
        num_parallel_calls=8)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size)
    
    dev_dataset = dev_dataset.map(_parse_function, \
        num_parallel_calls=8)
    dev_dataset = dev_dataset.batch(batch_size)
    dev_dataset = dev_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    test_dataset = test_dataset.map(_parse_function, \
        num_parallel_calls=8)
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

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
    json.dump({"user_vocab": user_id_map, "movie_vocab": movie_id_map, \
        "train_count": train_count, "dev_count": dev_count}, \
        open(os.path.join(data_dir, "datas.dump"), "w"))

    return train_count, dev_count, movie_id_map, user_id_map


def read_test_data(test_path, data_dir, movie_vocab, user_vocab):
    """
    """
    test_datas_count = 0
    test_tfpath = os.path.join(data_dir, "test.tfrecord")
    with tf.io.TFRecordWriter(test_tfpath) as test_tfwriter:
        with open(test_path, "r") as reader:
            movie_id = ""
            for line in reader:
                if ":" in line:
                    movie_id = line.strip().split(":")[0]
                else:
                    userid, timestamp = line.strip().split(",")
                    try:
                        sample = [movie_vocab[movie_id], user_vocab[userid], 0, timestamp]
                    except:
                        sample = [0, 0, 0, timestamp]
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
        'rating': tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
        }
    prased_example = tf.io.parse_single_example(example_proto, feature_description)

    return ({"movie_id": prased_example["movie_id"], \
        "user_id":prased_example["user_id"]}, prased_example["rating"])

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def agg(chunk):
    #chunk['timestamp'] = pd.to_datetime(chunk["timestamp"], format="%Y-%m-%d")
    grouped = chunk.groupby(["userid"])["timestamp"].mean(numeric_only=False)
    index = list(grouped.index)
    mean_vals = list([pd.Timestamp(val) for val in grouped.values])
    return zip(index, mean_vals)

def calculate_mean_date(chunks):
    """
    """
    pool = mlp.Pool(4)
    orphan = pd.DataFrame()
    result = []
    for chunk in chunks:
        chunk = pd.concat([orphan, chunk])
        last_key = chunk["userid"].iloc[-1]
        is_orphan = chunk["userid"] == last_key
        orphan = chunk[is_orphan]
        chunk = chunk[~is_orphan] 
        result.append(pool.apply_async(agg, args=(chunk, )))
    
    last_results = []
    for r in result:
        last_results.extend(list(r.get()))
    
    pool.close()
    pool.join()

    last_results.extend(agg(orphan))
    
    return dict(last_results)

def cal_item_timebias_wrap(timestamp, min_timestamp, interval_time):
    return timestamp.map(lambda x: math.ceil((x - min_timestamp).days / interval_time))
    
def cal_user_timedev_wrap(data, mean_data, beta):
    """
    """
    def dev_func(row):
        #print(row["timestamp"], user_mean_date_d[row["userid"]])
        value = math.pow(abs((row["timestamp"] - mean_data[row["userid"]]).days), beta)
        return round(value, 2) if row["timestamp"] > mean_data[row["userid"]] else -1 * round(value, 2)
    return data.apply(dev_func, axis=1)

def preprocess_df(data_path, path):
    """
    add the time dynamic
    """
    start_time = time.time()
    dataframe = pd.read_csv(data_path)
    print("---read csv time---", time.time() - start_time)
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], format="%Y-%m-%d")
    print("---to datetime---", time.time() - start_time)
    #dataframe.to_csv(tmp_path, index=False)
    # global stats
    min_timestamp = dataframe["timestamp"].min()
    max_timestamp = dataframe["timestamp"].max()
    delta_time = (max_timestamp - min_timestamp).days
    interval_time = math.ceil(delta_time / 30.0)
    print(min_timestamp, interval_time)

    # cal item time bias
    data_split = np.array_split(dataframe["timestamp"], 8)
    pool = mlp.Pool(8)
    func = partial(cal_item_timebias_wrap, min_timestamp=min_timestamp, interval_time=interval_time)
    dataframe["item_time_bias"] = pd.concat(pool.map(func ,data_split))
    pool.close()
    pool.join()
    print("---map item time bias---", time.time() - start_time)
    
    # cal the mean date of the user rating
    mean_data = calculate_mean_date(np.array_split(dataframe, 8))
    print("---global mean calculation---", time.time() - start_time)
    print("--dict length---", len(mean_data), len(dataframe["userid"].unique()))
    
    # cal the deviation
    beta = 0.4
    pool = mlp.Pool(8)
    data_split = np.array_split(dataframe, 8)
    dev_func_partial = partial(cal_user_timedev_wrap, mean_data=mean_data, beta=beta)
    dataframe["user_time_dev"] = pd.concat(pool.map(dev_func_partial, data_split))
    pool.close()
    pool.join()
    print("---user time dev---", time.time() - start_time)
    print(dataframe.head(2))
    dataframe.to_csv(path, index=False)
    return dataframe
    

if __name__ == "__main__":
    train_dir = "/home/lyt/workspace/recsys/data/netflixprize/training_set"
    dev_path = "/home/lyt/workspace/recsys/data/netflixprize/probe.txt"
    test_path = "/home/lyt/workspace/recsys/data/netflixprize/qualifying.txt"
    data_dir = "/home/lyt/workspace/recsys/tf_impl_reco/data/"
    
    #make_tfrecords(train_dir, dev_path, test_path, data_dir)
    start_time = time.time()
    # take 770s
    # read_traindev_arraydata(data_dir, train_dir, dev_path)
    print("-----read dataset time is ---", time.time() - start_time)
    
    train_path = os.path.join(data_dir, "train" + ".csv")
    train_processed_path = os.path.join(data_dir, "train_processed" + ".csv")
    #preprocess_df(train_path, train_processed_path)
    
    dev_path = os.path.join(data_dir, "dev" + ".csv")
    dev_processed_path = os.path.join(data_dir, "dev_processed" + ".csv")
    #preprocess_df(dev_path, dev_processed_path)
    
    batch_size = 64
    epochs = 5
    shuffle_len = 100000
    #make_netflix_dataset(data_dir, 147, batch_size, epochs)
    train_dataset, dev_dataset, train_len, dev_len = make_netflix_tensor_dataset(\
        data_dir, batch_size, epochs, shuffle_len)

    for batch in train_dataset.take(1):
        print(batch)
        break
