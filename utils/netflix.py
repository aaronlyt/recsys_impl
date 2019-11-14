"""
parse the netfloxprize dataset
"""
import sys
import os
import json
import numpy as np
import tensorflow as tf


def make_netflix_dataset(train_dir, dev_path, test_path, data_dir, \
    batch_size, epochs, cache=True):
    """
    """
    if cache and os.path.exists(os.path.join(data_dir, "datas.dump")):
        datas = json.load(open(os.path.join(data_dir, "datas.dump"), "r"))
        train_datas, dev_datas = datas["train"], datas["dev"]
        movie_vocab, user_vocab = datas["movie_vocab"], datas["user_vocab"]
    else:
        train_datas, dev_datas, movie_vocab, user_vocab = \
            read_traindev_data(train_dir, dev_path, data_dir, cache)
    test_datas = read_test_data(test_path, user_vocab, movie_vocab)

    train_dataset = tf.data.Dataset.from_tensor_slices(\
        ({"user_id": np.array([[sample[1]] for sample in train_datas]), \
            "item_id": np.array([[sample[0]] for sample in train_datas])}, \
            np.array([sample[2] for sample in train_datas])))
    dev_dataset = tf.data.Dataset.from_tensor_slices(\
        ({"user_id": np.array([[sample[1]] for sample in dev_datas]), \
            "item_id": np.array([[sample[0]] for sample in dev_datas])}, \
                np.array([sample[2] for sample in dev_datas])))
    dev_dataset = tf.data.Dataset.from_tensor_slices(\
        ({"user_id": np.array([[sample[1]] for sample in test_datas]), \
            "item_id": np.array([[sample[0]] for sample in test_datas])}, \
                np.array([sample[2] for sample in test_datas])))

    train_dataset = train_dataset.shuffle(len(train_datas))
    train_dataset = train_dataset.repeat(epochs)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(2)

    dev_dataset = dev_dataset.batch(batch_size)
    dev_dataset = dev_dataset.prefetch(2)

    test_dataset = dev_dataset.batch(batch_size)
    test_dataset = dev_dataset.prefetch(2)

    return train_dataset, dev_dataset, test_dataset, len(movie_vocab), len(user_vocab)


def read_traindev_data(train_dir, dev_path, data_dir, cache=True):
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
    
    train_datas = []
    ldev_datas = []
    user_id_map = {}
    movie_id_map = {}
    for filename in os.listdir(train_dir):
        filepath = os.path.join(train_dir, filename)
        with open(filepath, "r") as reader:
            movie_id = next(reader).strip().split(":")[0]
            movie_id_map.setdefault(movie_id, len(movie_id))
            for user_line in reader:
                userid, rating, timestamp = user_line.strip().split(",")
                user_id_map.setdefault(userid, len(user_id_map))
                sample = [movie_id_map[movie_id], user_id_map[userid], float(rating), timestamp]
                if "%s_%s" %(movie_id, userid) not in dev_datas:
                    train_datas.append(sample)     
                else:
                    ldev_datas.append(sample)
                
    print("---train dev dataset summary----", len(train_datas), len(ldev_datas))
    print("---vocab summary----", len(user_id_map), len(movie_id_map))
   
    if cache:
        json.dump({"train": train_datas, "dev": ldev_datas, \
            "user_vocab": user_id_map, "movie_vocab": user_id_map}, \
                open(os.path.join(data_dir, "datas.dump"), "w"))

    return train_datas, ldev_datas, movie_id_map, user_id_map


def read_test_data(test_path, movie_vocab, user_vocab):
    """
    """
    test_datas = {}
    with open(test_path, "r") as reader:
        movie_id = ""
        for line in reader:
            if ":" in line:
                movie_id = line.strip().split(":")[0]
            else:
                try:
                    userid, timestamp = line.strip().split(",")
                except:
                    print(line)
                #test_datas.append([\
                #    movie_vocab[movie_id], user_vocab[userid], 0, timestamp])
    return test_datas

if __name__ == "__main__":
    train_dir = "/home/lyt/workspace/recsys/data/netflixprize/training_set"
    dev_path = "/home/lyt/workspace/recsys/data/netflixprize/probe.txt"
    test_path = "/home/lyt/workspace/recsys/data/netflixprize/qualifying.txt"
    data_dir = "/home/lyt/workspace/recsys/tf_impl_reco/data/"
    #read_traindev_data(train_dir, dev_path, data_dir)

    batch_size = 64
    epochs = 5
    make_netflix_dataset(train_dir, dev_path, test_path, data_dir, batch_size, epochs, cache=True)