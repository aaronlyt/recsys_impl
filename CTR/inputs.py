import os
import sys
import json
import tensorflow as tf


def dense_trans_feat_func(min_val, max_val):
    def dense_trans_func(val):
        return (val - min_val) / (max_val - min_val)
    return dense_trans_func


def create_feature_columns(dataset_dir, sp_feat_dim, initializer=None, is_debug=True):
    """
    bucket size: categorical value appear more 10 times count, store in spfeat_stata.vocab
    """
    if not is_debug:
        vocab_path = os.path.join(dataset_dir, "spfeat_state.vocab")
    else:
        vocab_path = os.path.join(dataset_dir, "spfeat.vocab")
    vocab_info = json.load(open(vocab_path, "r"))

    dense_features = ["I" + str(idx) for idx in range(1, 14)]
    sparse_features = ["C" + str(idx) for idx in range(1, 27)]
    
    feature_columns = []
    for dense_feat in dense_features:
        feature_columns.append(tf.feature_column.numeric_column(dense_feat, \
                normalizer_fn=dense_trans_feat_func(vocab_info[dense_feat][0], vocab_info[dense_feat][1])))

    for sp_feat in sparse_features:
        if not is_debug:
            categorical_column_feat = tf.feature_column. \
                    categorical_column_with_hash_bucket(sp_feat, vocab_info[sp_feat])
        else:
            categorical_column_feat = tf.feature_column. \
                    categorical_column_with_vocabulary_list(sp_feat, vocab_info[sp_feat])
        
        feature_columns.append(tf.feature_column.embedding_column(\
            categorical_column_feat, sp_feat_dim, max_norm=None, initializer=initializer))
    return feature_columns
