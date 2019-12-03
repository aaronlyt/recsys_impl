"""
parse the netfloxprize dataset
"""
import sys
import os
import math
import datetime
import json
import multiprocessing as mlp
import numpy as np
import pandas as pd
import tensorflow as tf



def process_unit(filepath):
    """
    """
    train_dataset = []
    dev_dataset = []
    with open(filepath, "r") as reader:
        movie_id = next(reader).strip().split(":")[0]
        for user_line in reader:
            userid, rating, timestamp = user_line.strip().split(",")
            sample = [int(movie_id), int(userid), float(rating), \
                datetime.datetime.strptime(timestamp, "%Y-%m-%d")]
            train_dataset.append(sample)
            
    return train_dataset

def read_traindev_arraydata(data_dir, train_dir, dev_path):
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
    train_dataset = []
    dev_dataset = []
    pool = mlp.Pool(8)
    ret_jobs = []
    for filename in os.listdir(train_dir):
        filepath = os.path.join(train_dir, filename)
        ret_jobs.append((pool.apply_async(process_unit, args=(filepath,))))

    for job in ret_jobs:
        rets = job.get()
        train_dataset.extend(rets[0])
        dev_dataset.extend(rets[1])

    pool.join()

    train_count = len(train_dataset)
    dev_count = len(dev_dataset)
    print("-----train and dev dataset length----", train_count, dev_count)
    
    train_path = os.path.join(data_dir, "train" + ".csv")
    columns = ["movieid", "userid", "rating", "timestamp"]
    pd.DataFrame(train_dataset, columns=columns).to_csv(train_path, index=False, chunksize=1e7)

    dev_path = os.path.join(data_dir, "dev" + ".csv")
    pd.DataFrame(dev_dataset, columns=columns).to_csv(dev_path, index=False)

    return train_dataset, dev_dataset, train_count, dev_count


def make_netflix_tensor_dataset(data_dir, train_dir, dev_path, \
    batch_size, epochs, shuffle_len):
    """
    """
    train_dataset, dev_dataset, train_len, dev_len = \
        read_traindev_arraydata(data_dir, train_dir, dev_path)
    train_dataframe = pd.DataFrame(train_dataset, \
        columns=["movieId", "userId", "rating"])
    dev_dataframe = pd.DataFrame(dev_dataset, \
        columns=["movieId", "userId", "rating"])

    dataset = []
    train_dataset = tf.data.Dataset.from_tensor_slices(\
        ({"user_id": train_dataframe["userId"].values[: ,np.newaxis], \
            "movie_id": train_dataframe["movieId"].values[: ,np.newaxis]}, \
            train_dataframe["rating"].values))
    dev_dataset = tf.data.Dataset.from_tensor_slices(\
        ({"user_id": dev_dataframe["userId"].values[: ,np.newaxis], \
            "movie_id": dev_dataframe["movieId"].values[: ,np.newaxis]}, \
                dev_dataframe["rating"]))
    
    train_dataset = train_dataset.shuffle(shuffle_len)
    train_dataset = train_dataset.repeat(epochs)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(shuffle_len)
    
    dev_dataset = dev_dataset.batch(batch_size)

    return train_dataset, dev_dataset, train_len, dev_len

def agg(chunk):
    chunk['timestamp'] = pd.to_datetime(chunk["timestamp"], format="%Y-%m-%d")
    grouped = chunk.groupby(["userid"])["timestamp"].mean(numeric_only=False)
    index = list(grouped.index)
    mean_vals = list([pd.Timestamp(val) for val in grouped.values])
    return zip(index, mean_vals)


def calculate_mean_date(data_path):
    """
    """
    pool = mlp.Pool(4)
    chunks = pd.read_csv(data_path, chunksize=1e7)
    orphan = pd.DataFrame()
    result = []
    for chunk in chunks:
        chunk = pd.concat([orphan, chunk])
        last_key = chunk["userid"].iloc[-1]
        is_orphan = chunk["userid"] == last_key
        chunk = chunk[~is_orphan] 
        orphan = chunk[is_orphan]
        result.append(pool.apply_async(agg, args=(chunk, )))
    last_results = []
    for r in result:
        last_results.extend(list(r.get()))
    return last_results


def preprocess_df(dataset, path, tmp_path):
    """
    add the time dynamic
    """
    dataframe = pd.DataFrame(dataset, columns=["movieid", "userid", "rating", "timestamp"])
    dataframe.sort_values(by=["timestamp"], inplace=True)
    dataframe.to_csv(tmp_path, index=False)
    # global stats
    min_timestamp = dataframe.iloc[0, 3]
    max_timestamp = dataframe.iloc[-1, 3]
    delta_time = (max_timestamp - min_timestamp).days
    interval_time = math.ceil(delta_time / 30.0)
    dataframe["item_time_b"] = dataframe["timestamp"].map(lambda tm: (tm - min_timestamp).days // delta_time)
    # cal the mean date of the user rating
    mean_data = dataframe.groupby(["userid"])["timestamp"].mean(numeric_only=False)
    # cal the deviation
    beta = 0.4
    def dev_func(row):
        #print(row["timestamp"], user_mean_date_d[row["userid"]])
        value = math.pow(abs((row["timestamp"] - mean_data.loc[row["userid"]]).days), beta)
        return value if row["timestamp"] > mean_data.loc[row["userid"]] else -1 * value
    dataframe["user_time_dev"] = dataframe.apply(dev_func, axis=1)
    dataframe.to_csv(path, index=False)
    # user date bias index
    return dataframe
    


if __name__ == "__main__":
    train_dir = "/home/lyt/workspace/recsys/data/netflixprize/training_set"
    dev_path = "/home/lyt/workspace/recsys/data/netflixprize/probe.txt"
    test_path = "/home/lyt/workspace/recsys/data/netflixprize/qualifying.txt"
    data_dir = "/home/lyt/workspace/recsys/tf_impl_reco/data/"
    
    #make_tfrecords(train_dir, dev_path, test_path, data_dir)
    read_traindev_arraydata(data_dir, train_dir, dev_path)

    sys.exit(0)
    batch_size = 64
    epochs = 5

    train_dataset, dev_dataset, train_len, dev_len = \
            read_traindev_arraydata(data_dir, train_dir, dev_path)
    
    train_processed_path = os.path.join(data_dir, "train_processed" + ".csv")
    tmp_path = os.path.join(data_dir, "train_raw_d" + ".csv")
    preprocess_df(train_dataset, train_processed_path, tmp_path)

    dev_processed_path = os.path.join(data_dir, "dev_processed" + ".csv")
    tmp_path = os.path.join(data_dir, "dev_raw_d" + ".csv")
    preprocess_df(dev_dataset, dev_processed_path, tmp_path)