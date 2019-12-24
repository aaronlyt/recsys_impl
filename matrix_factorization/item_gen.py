"""
item based callaborate filterinig algorithm
Amazon.com Recommendations Item-to-Item Collaborative Filtering

problem:
    consuming time, suspending?
    rating store as: 1e9 dict?
"""

import os
import sys
import pickle
import time
import collections
import functools
import numpy as np
import scipy as sp
import pandas as pd
import multiprocessing as mlpc
import sklearn.preprocessing as pp
from tqdm import tqdm

import IPython

def cosine_similarities(mat):
    row_normed_mat = pp.normalize(mat.tocsr(), axis=0)
    return row_normed_mat * row_normed_mat.T


def group_func():
    """
    """
    pass


def chunk_process(chunks, key):
    """
    list of dataframe
    """
    orphan = pd.DataFrame()
    result = []
    pool = mlpc.Pool(8)
    for chunk in chunks:
        chunk = pd.concat([orphan, chunk], axis=1)

        last_key = chunks.iloc[-1, key]
        is_orphan = (chunk[key] == last_key)

        orphan = chunk[is_orphan]
        chunk = chunk[~is_orphan]
        pool.apply_async(group_func, args=(chunk, ))
    pass


def offline_cal_smi(train_path, meta_data_path, mrating_path, score_path):
    """
    @train_path: csv file
    read train file and calculate the item similarity
        item feature matrix
        user-items index
        item-users index
    """
    s_time = time.time()
    dataset = pd.read_csv(train_path)
    print("---read csv data time is----", time.time() - s_time)

    s_time = time.time()
    item_idx_mapping = {}
    user_idx_mapping = {}

    movieid_unique = list(dataset["movieid"].unique())
    item_idx_mapping = dict(zip(movieid_unique, range(len(movieid_unique))))

    userid_unique = list(dataset["userid"].unique())
    user_idx_mapping = dict(zip(userid_unique, range(len(userid_unique))))
    print("---meta data mapping time is----", time.time() - s_time)
    s_time = time.time()
    
    user_items_mapping = collections.defaultdict(list)
    item_users_mapping = collections.defaultdict(list)
    
    row_ind = dataset["movieid"].map(lambda x: item_idx_mapping[x])
    col_ind = dataset["userid"].map(lambda x: user_idx_mapping[x])
    # wasted 3155.946124315262s
    for idx, itemid in enumerate(row_ind):
        item_users_mapping[itemid].append(col_ind[idx])
        user_items_mapping[col_ind[idx]].append(itemid)

    pickle.dump({"user_idx_mapping": user_idx_mapping, "item_idx_mapping": item_idx_mapping, \
        "user_items_mapping": user_items_mapping, "item_users_mapping": item_users_mapping},  \
            open(meta_data_path, "wb"))
    print("---meta data cal time is----", time.time() - s_time)
    
    s_time = time.time()
    # sparse matrix construction,(item_count, user_count)
    data = dataset["rating"].values
    
    item_feats_matrix = sp.sparse.csr_matrix((data, (row_ind, col_ind)), \
        shape=(len(movieid_unique), len(userid_unique)))
    print("---user count:%d, item count: %d" %(len(user_idx_mapping), len(item_idx_mapping)))
    #np.save(mrating_path, item_feats_matrix)
    sp.sparse.save_npz(mrating_path, item_feats_matrix)

    score_matrix = cosine_similarities(item_feats_matrix)
    #np.save(score_path, score_matrix)
    sp.sparse.save_npz(score_path, score_matrix)

    print("---score cal time is----", time.time() - s_time)
    
    return True, "OK"


def cal_rating(user_items_mapping, raw_rating, sims_mat, pair_mat):
    """
    """
    score_results = []
    for pair in tqdm(pair_mat, total=len(pair_mat)):
        userid, itemid = pair[0], pair[1]
        score = 0.0
        cnt = len(user_items_mapping[userid])
        u_id_ratings = raw_rating.getcol(userid)
        item_id_sim = sims_mat.getcol(itemid)
        score = (u_id_ratings.T * item_id_sim).data[0]
        
        score_results.append((userid, itemid, score / cnt))
    """
    score_lt = []
    for pair in pair_mat:
        s_time = time.time()
        userid, itemid = pair[0], pair[1]
        score = 0.0
        cnt = 0
        for item_b in user_items_mapping[userid]:
            if item_b == itemid:
                continue
            rating = raw_rating.getrow(itemid).getcol(userid)[0][0]
            sim_val = sim_mat.getrow(itemid).getcol(item_b)[0][0]
            score += rating * sim_val
            cnt += 1
        score_lt.append((userid, itemid, score / cnt))
    """
    return score_results

def validation_dev(meta_data_path, mrating_path, score_path, dev_path):
    """
    predict and evaluation
    """
    meta_data = pickle.load(open(meta_data_path, "rb"))
    item_users_mapping = meta_data["item_users_mapping"]
    user_items_mapping = meta_data["user_items_mapping"]
    user_idx_mapping = meta_data["user_idx_mapping"]
    item_idx_mapping = meta_data["item_idx_mapping"]

    #raw_rating = np.load(mrating_path, allow_pickle=True)
    #sims_mat = np.load(score_path, allow_pickle=True)
    raw_rating = sp.sparse.load_npz(mrating_path)
    sims_mat = sp.sparse.load_npz(score_path)

    print("---load data done----")
    
    #IPython.embed()

    dev_dataset = pd.read_csv(dev_path)
    userids = dev_dataset["userid"].map(lambda x: user_idx_mapping[x])
    itemids = dev_dataset["movieid"].map(lambda x: item_idx_mapping[x])
    indices_dataset = np.stack([userids, itemids], axis=1)
    dev_dataset_splits = np.array_split(indices_dataset, 6)
    # for loop calculating, for test
    print("---being to calculate----")
    """ 
    pool = mlpc.Pool(6)
    pool_rets = []
    partial_cal = functools.partial(cal_rating, user_items_mapping, raw_rating, sims_mat)
    for data in dev_dataset_splits:
        pool_rets.append(pool.apply_async(partial_cal, args=(data, )))
    
    score_results = []
    userid_rt = []
    itemid_rt = []
    for r in pool_rets:
        pool_rt = list(r.get())
        userid_rt.extend([val[0] for val in pool_rt])
        itemid_rt.extend([val[1] for val in pool_rt])
        score_results.extend([val[2] for val in pool_rt])
    
    pool.close()
    pool.join()

    assert(userids == userid_rt and itemids == itemid_rt)
    """
    score_results = []
    for pair in tqdm(indices_dataset, total=len(indices_dataset)):
        userid, itemid = pair[0], pair[1]
        score = 0.0
        cnt = len(user_items_mapping[userid])
        # sparse matrix index waste time
        u_id_ratings = raw_rating.getcol(userid)
        item_id_sim = sims_mat.getcol(itemid)
        score = (u_id_ratings.T * item_id_sim).data[0]
        
        score_results.append(score / cnt if score/cnt <= 5.0 else 5.0)
        
    mse = np.sqrt(np.sum(np.square(score_results - dev_dataset["rating"].values)))

    print("---mse result is----", mse)

if __name__ == "__main__":
    train_path = "/home/lyt/workspace/recsys/tf_impl_reco/data/train.csv"
    meta_data_path = "/home/lyt/workspace/recsys/tf_impl_reco/data/itemcfmeta.json"
    mrating_path = "/home/lyt/workspace/recsys/tf_impl_reco/data/rawrating.npz"
    mscore_path = "/home/lyt/workspace/recsys/tf_impl_reco/data/scores.npz"
    #offline_cal_smi(train_path, meta_data_path, mrating_path, mscore_path)

    dev_path = "/home/lyt/workspace/recsys/tf_impl_reco/data/dev.csv"
    validation_dev(meta_data_path, mrating_path, mscore_path, dev_path)
