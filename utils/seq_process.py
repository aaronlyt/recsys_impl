"""
tackle sequence dataset for prediction
"""
"""
Classes describing datasets of user-item interactions. Instances of these
are returned by dataset-fetching and dataset-processing functions.
"""

import os
import sys
import collections
import numpy as np
import scipy.sparse as sp


def _sliding_window(tensor, window_size, step_size=1):

    for i in range(len(tensor), 0, -step_size):
        yield tensor[max(i - window_size, 0):i]


def _generate_sequences(user_ids, item_ids,
                        indices,
                        max_sequence_length,
                        step_size):

    for i in range(len(indices)):
        #new user appear
        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]
        # sliding the user_i's items sequences
        for seq in _sliding_window(item_ids[start_idx:stop_idx],
                                   max_sequence_length,
                                   step_size):

            yield (user_ids[i], seq)


class Interactions(object):
    """
    Interactions object. Contains (at a minimum) pair of user-item
    interactions, but can also be enriched with ratings, timestamps,
    and interaction weights.

    For *implicit feedback* scenarios, user ids and item ids should
    only be provided for user-item pairs where an interaction was
    observed. All pairs that are not provided are treated as missing
    observations, and often interpreted as (implicit) negative
    signals.

    For *explicit feedback* scenarios, user ids, item ids, and
    ratings should be provided for all user-item-rating triplets
    that were observed in the dataset.

    Parameters
    ----------

    user_ids: array of np.int32
        array of user ids of the user-item pairs
    item_ids: array of np.int32
        array of item ids of the user-item pairs
    ratings: array of np.float32, optional
        array of ratings
    timestamps: array of np.int32, optional
        array of timestamps
    weights: array of np.float32, optional
        array of weights
    num_users: int, optional
        Number of distinct users in the dataset.
        Must be larger than the maximum user id
        in user_ids.
    num_items: int, optional
        Number of distinct items in the dataset.
        Must be larger than the maximum item id
        in item_ids.

    Attributes
    ----------

    user_ids: array of np.int32
        array of user ids of the user-item pairs
    item_ids: array of np.int32
        array of item ids of the user-item pairs
    ratings: array of np.float32, optional
        array of ratings
    timestamps: array of np.int32, optional
        array of timestamps
    weights: array of np.float32, optional
        array of weights
    num_users: int, optional
        Number of distinct users in the dataset.
    num_items: int, optional
        Number of distinct items in the dataset.
    """

    def __init__(self, user_ids, item_ids,
                 ratings=None,
                 timestamps=None,
                 weights=None,
                 num_users=None,
                 num_items=None):

        self.num_users = num_users or int(user_ids.max()) + 1
        # item start from one
        self.num_items = num_items or int(item_ids.max()) + 1

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.timestamps = timestamps
        self.weights = weights

        item_counter = [val[1] for val in \
            sorted(collections.Counter(item_ids).items(), key=lambda x: x[0])]
        self.item_frequency = np.array(item_counter) / sum(item_counter)
        self.sampling_count = 1
        self.pred_seq = True

    def to_sequence(self, max_sequence_length=10, min_sequence_length=None, step_size=None):
        """
        Transform to sequence form.

        User-item interaction pairs are sorted by their timestamps,
        and sequences of up to max_sequence_length events are arranged
        into a (zero-padded from the left) matrix with dimensions
        (num_sequences x max_sequence_length).

        Valid subsequences of users' interactions are returned. For
        example, if a user interacted with items [1, 2, 3, 4, 5], the
        returned interactions matrix at sequence length 5 and step size
        1 will be be given by:

        .. code-block:: python

           [[1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4],
            [0, 0, 1, 2, 3],
            [0, 0, 0, 1, 2],
            [0, 0, 0, 0, 1]]

        At step size 2:

        .. code-block:: python

           [[1, 2, 3, 4, 5],
            [0, 0, 1, 2, 3],
            [0, 0, 0, 0, 1]]

        Parameters
        ----------

        max_sequence_length: int, optional
            Maximum sequence length. Subsequences shorter than this
            will be left-padded with zeros.
        min_sequence_length: int, optional
            If set, only sequences with at least min_sequence_length
            non-padding elements will be returned.
        step-size: int, optional
            The returned subsequences are the effect of moving a
            a sliding window over the input. This parameter
            governs the stride of that window. Increasing it will
            result in fewer subsequences being returned.

        Returns
        -------

        sequence interactions: :class:`~SequenceInteractions`
            The resulting sequence interactions.
        """

        if self.timestamps is None:
            raise ValueError('Cannot convert to sequences, '
                             'timestamps not available.')

        if 0 in self.item_ids:
            raise ValueError('0 is used as an item id, conflicting '
                             'with the sequence padding value.')

        if step_size is None:
            step_size = max_sequence_length

        # Sort first by user id, then by timestamp
        sort_indices = np.lexsort((self.timestamps,
                                   self.user_ids))

        user_ids = self.user_ids[sort_indices]
        item_ids = self.item_ids[sort_indices]
        
        user_ids, indices, counts = np.unique(user_ids,
                                              return_index=True,
                                              return_counts=True)

        num_subsequences = int(np.ceil(counts / float(step_size)).sum())

        sequences = np.zeros((num_subsequences, max_sequence_length),
                             dtype=np.int32)
        sequence_users = np.zeros(num_subsequences, dtype=np.int32)
        pred_sequence_len = max_sequence_length - 1
        sequence_targets = np.zeros((num_subsequences, pred_sequence_len), dtype=np.int32)
        sequence_negs = np.zeros((num_subsequences, pred_sequence_len, \
            self.sampling_count), dtype=np.int32)
        for i, (uid,
                seq) in enumerate(_generate_sequences(user_ids,
                                                      item_ids,
                                                      indices,
                                                      max_sequence_length,
                                                      step_size)):
            assert(max(seq) < self.num_items)
            if len(seq) <= 1:
                continue
            sequence_users[i] = uid
            real_seq_len = len(seq)
            real_seq_pred_len = real_seq_len - 1
            if self.pred_seq:
                sequences[i][:real_seq_len] = seq
                sequence_targets[i][:real_seq_pred_len] = seq[1:]
            else:
                sequences[i][:real_seq_pred_len] = seq[:-1]
                sequence_targets[i] = seq[-1:]
            # an array
            neg_samples = np.random.choice(self.num_items - 1, \
                    (real_seq_pred_len, self.sampling_count), p=self.item_frequency) + 1
            assert(np.max(neg_samples) < self.num_items)
            sequence_negs[i][:real_seq_pred_len, :] = neg_samples

        if min_sequence_length is not None:
            long_enough = sequences[:, min_sequence_length] != 0
            sequences = sequences[long_enough ]
            sequence_users = sequence_users[long_enough]
            sequence_targets = sequence_targets[long_enough]
            sequence_negs = sequence_negs[long_enough]

        return sequence_users, sequence_targets, sequence_negs, sequences


if __name__ == "__main__":
    import pandas as pd
    data_path = "/home/lyt/workspace/recsys/data/ml-20m/ratings.csv"
    dtype_dict = {"userId":np.int32, "movieId": np.int32, "rating":np.float32, \
        "timestamp":np.int32}
    raw_data = pd.read_csv(data_path, dtype=dtype_dict)
    dataset = Interactions(raw_data["userId"].values, \
        raw_data["movieId"].values, raw_data["rating"].values, \
            raw_data["timestamp"].values)
    seq_dataset = dataset.to_sequence()
    
    for idx in range(seq_dataset.max_sample_count):
        print("--------sequence----------", seq_dataset.sequences[idx])
        print("----user id and target-------", seq_dataset.user_ids[idx], seq_dataset.target_ids[idx])

        if idx >= 10:
            break
