"""
sequence to sequence prediction
"""
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class DNNModel(keras.Model):
    def __init__(self, item_count, item_emb_dim, mlp_units, dropout_rate):
        """
        @param args
        """
        super(DNNModel, self).__init__()
        self.item_embeddings = keras.layers.Embedding(item_count, \
            item_emb_dim, mask_zero=True)
        self.dense_layers = []
        for units in mlp_units:
            self.dense_layers.append(keras.layers.Dense(units, activation="relu"))
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, sequence_ids, target_id=None, training=True):
        """
        do not predict the first item
        @param sequence_ids, bs_size, seq_len
        @param target_id, are sequences in traning, (bs_size, seq_len, sample_count)
        """
        sequence_embs = self.item_embeddings(sequence_ids)
        # bs_size, seq_len - 1, emb_dim, exclude last sum, remove zero
        user_history_repr = tf.math.cumsum(sequence_embs, axis=1, exclusive=True)[:, 1:]
        for dense_layer in self.dense_layers:
            user_history_repr = dense_layer(user_history_repr)
            if training:
                user_history_repr = self.dropout(user_history_repr, training=training)
        # cal score
        if training:
            # bs_size, seq_len - 1, sample_count, dim
            if len(target_id.shape) == 2:
                target_id = tf.expand_dims(target_id, axis=2)
            target_emb = self.item_embeddings(target_id)
        else:
            # 1, 1, item_count, dim
            target_emb = tf.expand_dims(self.item_embeddings.embeddings[1:,:], axis=0)
            target_emb = tf.expand_dims(target_emb, axis=1)
        user_history_repr = tf.expand_dims(user_history_repr, axis=2)
        # bs_size, seq_len - 1, sample_count
        scores = tf.matmul(user_history_repr, target_emb, transpose_b=True)
        #print(scores)
        return tf.squeeze(scores), tf.squeeze(sequence_embs._keras_mask[:, :-1])


class LSTMSeqModel(keras.Model):
    def __init__(self, item_count, item_emb_dim, attention=False):
        """
        """
        super(LSTMSeqModel, self).__init__()
        self.item_embeddings = keras.layers.Embedding(item_count, item_emb_dim, mask_zero=False)
        self.lstm = keras.layers.LSTM(item_emb_dim, return_sequences=True)
        self.attention = attention
    
    def call(self, sequence_ids, target_id=None, training=True):
        """
        @param sequence_ids, bs_size, seq_len
        @param target_id, are sequences in traning, (bs_size, seq_len, sample_count)
        """
        sequence_emb = self.item_embeddings(sequence_ids)
        lstm_seq_repr = self.lstm(sequence_emb[:, :-1])

        if training:
            if len(target_emb.shape) == 2:
                target_emb = tf.expand_dims(target_emb, axis=2)
            # target_emb, bs_size, seq_len, sample_count, dim
            target_emb = self.item_embeddings(target_id)
        else:
            # ignore padding item, item_count, emb_dim
            target_emb = self.item_embeddings.embeddings[1:, :]
            target_emb = tf.expand_dims(target_emb, axis=0)
            target_emb = tf.exand_dims(target_emb, axis=1)
        # bs_size, item_count(training is 1,  pred is item_total_count)
        if not self.attention:
            user_repr = lstm_seq_repr
            # bs_size, seq_len - 1, sample_count
            score = tf.matmul(user_repr, target_emb, transpose_b=True)
        else:
            score = self.cal_pred_attention_score(lstm_seq_repr, target_emb)

        return tf.squeeze(score)

    def cal_pred_attention_score(self, sequence_emb, item_emb):
        """
        @param sequence_emb: bs_size, seq_len, emb_dim
        @param target_emb: (bsize, seq_len, sample_count, emb_dim)
        return: score logits, bs_size, item_count
        """
        #bs_size, 1, seq_len, emb_dim
        sequence_emb = tf.expand_dims(sequence_emb, axis=1)
        # bs_size, seq_len, item_count, seq_len
        sim_logits = tf.matmul(item_emb, sequence_emb, transpose_b=True)
        weight = tf.math.softmax(sim_logits, axis=-1)
        # user repr, bs_size, seq_len, item_count, emb_dim
        user_repr = tf.matmul(weight, sequence_emb, transpose_a=True)
        pred_logits = tf.math.reduce_sum(user_repr * item_emb, axis=-1)
        return pred_logits
