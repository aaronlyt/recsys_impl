import os
import sys
import json
import collections
import pandas as pd
import numpy as np


def prepare_input_data(dataset_dir, dataset_path, dev_size=0.1):
    """
    split train and dev dataset, get the data count
    make sp_features vocab
    """
    tra_path = os.path.join(dataset_dir, "criteo.tra")
    dev_path = os.path.join(dataset_dir, "criteo.dev")
    feat_vocab_path = os.path.join(dataset_dir, "spfeat.vocab")
    
    if os.path.isfile(tra_path):
        os.remove(tra_path)
    if os.path.isfile(dev_path):
        os.remove(dev_path)

    dataset = pd.read_csv(dataset_path, sep="\t", \
        chunksize=1000000, header=None)
    dense_features = ["I" + str(idx) for idx in range(1, 14)]
    sparse_features = ["C" + str(idx) for idx in range(1, 27)]
    columns = ["label"] + dense_features + sparse_features
    # dict of list
    vocab_dict = collections.defaultdict(set)
    train_count = 0
    dev_count = 0
    with open(tra_path, "a+") as tra_writer, \
        open(dev_path, "a+") as dev_writer:
            tra_writer.write(",".join(columns)+"\n")
            dev_writer.write(",".join(columns)+"\n")
            for chunk_data in dataset:
                chunk_data.columns = columns
                chunk_data[dense_features] = chunk_data[dense_features].fillna(0.0)
                chunk_data[sparse_features] = chunk_data[sparse_features].fillna('masked_nan_val')
                for ds_feat in dense_features:
                    min_val = min(float(chunk_data[ds_feat].min()), vocab_dict[ds_feat][0] \
                        if ds_feat in vocab_dict else 1e8)
                    max_val = max(float(chunk_data[ds_feat].max()), vocab_dict[ds_feat][1] \
                        if ds_feat in vocab_dict else -10000)
                    vocab_dict[ds_feat] = [min_val, max_val]
                for sp_feat in sparse_features:
                    vocab_dict[sp_feat].update(set(chunk_data[sp_feat].unique()))
                # write to files
                val = np.random.choice(2, p=[dev_size, 1 - dev_size])
                if val == 1:
                    chunk_data.to_csv(tra_writer, header=False, index=False)
                    train_count += chunk_data.shape[0]
                elif val == 0:
                    chunk_data.to_csv(dev_writer, header=False, index=False)
                    dev_count += chunk_data.shape[0]
                print("---count---", train_count, dev_count)
    for key in vocab_dict:
        vocab_dict[key] = list(vocab_dict[key])
    vocab_dict["train_count"] = train_count
    vocab_dict["dev_count"] = dev_count
    json.dump(vocab_dict, open(feat_vocab_path, "w"))


def prepare_feats_data(dataset_dir, dataset_path, dev_size=0.1):
    """
    split train and dev dataset, get the data count
    make sp_features vocab
    should stat the min max value
    """
    feat_vocab_path = os.path.join(dataset_dir, "spfeat_state.vocab")
    dataset = pd.read_csv(dataset_path, sep="\t", chunksize=1000000, header=None)
    dense_features = ["I" + str(idx) for idx in range(1, 14)]
    sparse_features = ["C" + str(idx) for idx in range(1, 27)]
    columns = ["label"] + dense_features + sparse_features
    # dict of list
    vocab_states_dict = collections.defaultdict(dict)
    train_count = 0
    dev_count = 0
    for chunk_data in dataset:
        chunk_data.columns = columns
        chunk_data[dense_features] = chunk_data[dense_features].fillna(0.0)
        chunk_data[sparse_features] = chunk_data[sparse_features].fillna('masked_nan_val')
        for ds_feat in dense_features:
            min_val = min(float(chunk_data[ds_feat].min()), vocab_states_dict[ds_feat].get(\
                "min", 1e8))
            max_val = max(float(chunk_data[ds_feat].max()), vocab_states_dict[ds_feat].get(\
                "max", -1e8))
            vocab_states_dict[ds_feat]["min"] = min_val
            vocab_states_dict[ds_feat]["max"] = max_val
        for sp_feat in sparse_features:
            [vocab_states_dict[sp_feat].update({val: vocab_states_dict[sp_feat].get(val, 0) + 1}) \
                for val in chunk_data[sp_feat].values]
        print("--processed chunksize--", chunk_data.shape[0])
    vocab_states_dict_main = {}
    for key in sparse_features:
        greater_than_10 = len(list(filter(lambda x: x[1] >= 2, vocab_states_dict[key].items())))
        print("-----key, count, count greater then 10 is----", key, len(vocab_states_dict[key]), greater_than_10)
        vocab_states_dict_main[key] = greater_than_10
    
    for key in dense_features:
        vocab_states_dict_main[key] = [vocab_states_dict[key]["min"], vocab_states_dict[key]["max"]]
    
    vocab_states_dict_main["train_count"] = train_count
    vocab_states_dict_main["dev_count"] = dev_count

    json.dump(vocab_states_dict_main, open(feat_vocab_path, "w"))

    return True, "OK"


if __name__ == "__main__":
    dataset_dir = "/home/lyt/workspace/recsys/data/criteo_data/"
    dataset_path = "/home/lyt/workspace/recsys/data/criteo_data/train.txt"
    prepare_input_data(dataset_dir, dataset_path, dev_size=0.1)
    prepare_feats_data(dataset_dir, dataset_path)

    debug_dataset_dir = "/home/lyt/workspace/recsys/data/criteo_data/debug_data/"
    debug_dataset_path = "/home/lyt/workspace/recsys/data/criteo_data/debug_data/train.txt"
    #prepare_input_data(debug_dataset_dir, debug_dataset_path, dev_size=0.1)

