import os
import sys
import itertools
import time
import pandas as pd
import numpy as np
import tensorflow as tf

sys.path.append('../../')

from tf_impl_reco.utils.seq_process import *

def _parse_line(line, separator='::'):

    uid, iid, rating, timestamp = line.split(separator)

    return (int(uid), int(iid), float(rating), int(timestamp))


def _make_contiguous(data, separator):

    user_map = {}
    item_map = {}

    for line in data:
        uid, iid, rating, timestamp = _parse_line(line, separator=separator)

        uid = user_map.setdefault(uid, len(user_map) + 1)
        iid = item_map.setdefault(iid, len(item_map) + 1)

        yield uid, iid, rating, timestamp

def make_df_dataset():
    """
    """
    filepath = "/home/lyt/workspace/recsys/data/ml-10M100K/ratings.dat"
    to_filepath = "/home/lyt/workspace/recsys/data/ml-10M100K/ratings.csv"
    dataset = []
    with open(filepath, "r") as reader:
        for line in reader:
            userid, movieid, rating, timestamp = line.strip().split("::")
            dataset.append([userid, movieid, rating, timestamp])
    dataframe = pd.DataFrame(dataset, \
        columns=["userId", "movieId", "rating", "timestamp"])
    print("---total dataset length------", len(dataframe))
    dataframe.to_csv(to_filepath, index=False)
    return True, "OK"


def read_movielens_20M(filepath, batch_size, epochs, train_ratio=0.8):
    dataframe = pd.read_csv(filepath)
    dataframe["userId"] = dataframe["userId"].astype("int32")
    dataframe["movieId"] = dataframe["movieId"].astype("int32")
    dataframe["rating"] = dataframe["rating"].astype("float32")

    dataset_len = len(dataframe)
    dataframe = dataframe.iloc[np.random.permutation(dataset_len)]
    train_len = int(dataset_len * train_ratio)
    train_dataframe = dataframe.iloc[:train_len, :]
    dev_dataframe = dataframe.iloc[train_len:, :]

    train_dataset = tf.data.Dataset.from_tensor_slices(\
        ({"user_id": train_dataframe["userId"].values[: ,np.newaxis], \
            "item_id": train_dataframe["movieId"].values[: ,np.newaxis]}, \
            train_dataframe["rating"].values))
    dev_dataset = tf.data.Dataset.from_tensor_slices(\
        ({"user_id": dev_dataframe["userId"].values[: ,np.newaxis], \
            "item_id": dev_dataframe["movieId"].values[: ,np.newaxis]}, \
                dev_dataframe["rating"]))
    train_dataset = train_dataset.shuffle(train_len)
    train_dataset = train_dataset.repeat(epochs)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(2)
    
    dev_dataset = dev_dataset.batch(batch_size)
    dev_dataset = dev_dataset.prefetch(2)

    user_count = max(dataframe["userId"])
    movie_count = max(dataframe["movieId"])
    
    return train_dataset, dev_dataset, user_count, movie_count


def make_movielens_seqs_dataset(filepath, batch_size, dev_batch_size,epochs, train_ratio=0.9):
    """
    """
    start_time = time.time()
    dtype_dict = {"userId":np.int32, "movieId": np.int32, "rating":np.float32, \
        "timestamp":np.int32}
    dataframe = pd.read_csv(filepath, dtype=dtype_dict).head(1000)
    # user id mapping and movie id mapping
    userId_mapping = dict(\
        zip(dataframe["userId"].unique(), range(len(dataframe["userId"].unique())))\
            )
    movieId_mapping = dict(\
        zip(dataframe["movieId"].unique(), range(1, len(dataframe["movieId"].unique()) + 1))\
            )
    movieId_mapping[0] = 0

    dataframe["userId"] = dataframe["userId"].map(lambda x: userId_mapping[x])
    dataframe["movieId"] = dataframe["movieId"].map(lambda x: movieId_mapping[x])
    print("----read and transform datafram done---", time.time() - start_time)
    # get sequences
    inter_obj = Interactions(dataframe["userId"].values, dataframe["movieId"].values, \
        dataframe["rating"].values, dataframe["timestamp"].values)
    sequence_users, sequence_targets, sequence_negs, sequences = inter_obj.to_sequence(step_size=2)
    print("----to sequences done---", time.time() - start_time)
    
    dataset_len = len(sequence_users)
    indices = np.random.permutation(range(dataset_len))
    train_len = int(dataset_len * train_ratio)
    dev_len = dataset_len - train_len

    train_dataset = tf.data.Dataset.from_tensor_slices(\
        ({"user_id": sequence_users[indices[:train_len]], \
            "sequence": sequences[indices[:train_len]], \
                "neg_id": sequence_negs[indices[:train_len]]}, \
                    sequence_targets[indices[:train_len]]))
    dev_dataset = tf.data.Dataset.from_tensor_slices(\
        ({"user_id": sequence_users[indices[train_len:]], \
            "sequence": sequences[indices[train_len:]]}, \
            sequence_targets[indices[train_len:]]))
    
    train_dataset = train_dataset.shuffle(train_len)
    train_dataset = train_dataset.repeat(epochs)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    dev_dataset = dev_dataset.batch(dev_batch_size)
    dev_dataset = dev_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    user_count = len(userId_mapping)
    movie_count = len(movieId_mapping)

    return train_dataset, dev_dataset, train_len, dev_len, user_count, movie_count


if __name__ == "__main__":
    #read_movielens_20M("/home/lyt/workspace/recsys/data/ml-20m/ratings.csv")
    #make_df_dataset()
    train_data, dev_data, train_len, dev_len, user_count, movie_coutn = \
        make_movielens_seqs_dataset("/home/lyt/workspace/recsys/data/ml-20m/ratings.csv", 64, 2)
    
    for batch in train_data.take(1):
        print(batch)

    print(train_len, dev_len, user_count, movie_coutn)