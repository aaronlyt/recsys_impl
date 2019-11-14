import itertools
import os
import pandas as pd
import numpy as np
import tensorflow as tf

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


if __name__ == "__main__":
    tf.enable_eager_execution()
    #read_movielens_20M("/home/lyt/workspace/recsys/data/ml-20m/ratings.csv")
    make_df_dataset()