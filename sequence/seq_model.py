"""
the train predict wrap class
"""
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tf_impl_reco.sequence.repr_layer import *
from tf_impl_reco.sequence.losses import *
from tf_impl_reco.utils.metrics import *


class SeqModel(object):
    def __init__(self, config):
        """
        @param config, params dict, best loads from file
        """
        self.config = config
        self._build_model(config)
        
    def _build_model(self, config):
        """
        """
        if config.repr == "dnn":
            self._net = DNNModel(config.item_count, config.item_emb_dim, \
                config.mlp_units, config.dropout_rate)
        elif config.repr == "lstm":
            self._net = LSTMSeqModel(config.item_count, config.item_emb_dim)
        elif config.repr == "lstm_att":
            self._net = LSTMSeqModel(config.item_count, config.item_emb_dim, attention=True)

        if config.loss_func == 'pointwise':
            self._loss_func = pointwise_loss
        elif config.loss_func == 'bpr':
            self._loss_func = bpr_loss
        elif config.loss_func == 'hinge':
            self._loss_func = hinge_loss
            self._predict_batch = self._predict_batch_hinge
        else:
            self._loss_func = adaptive_hinge_loss
        
        if config.optimizer is None:
            self._optimizer = keras.optimizers.Adam(config.learning_rate)

    def fit(self, args, train_data, dev_data):
        """
        """
        for epoch in range(args.epochs):
            for batch_data in tqdm(train_data, total=args.steps_per_epoch):
                loss = self._fit_batch(batch_data)
            print(loss)

            # at the condition, accuracy@N is the recall@N, and is the hits@N
            m_acc = tf.metrics.SparseTopKCategoricalAccuracy(k=20)
            m = NDGG(k=20)
            for batch_data, label in tqdm(dev_data, total=args.val_steps):
                # bs_size, num_items
                prediction = self._predict_batch(batch_data)
                m.update_state(label - 1, prediction)
                m_acc.update_state(label - 1, prediction)
            print("---epoch: %d, auc:%.4f, ncg:%.4f" %(epoch, m_acc.result(), \
                m.result()))

    def _fit_batch(self, batch_data):
        """
        """
        inputs, target_id = batch_data
        sequence_ids, neg_id = inputs["sequence"], inputs["neg_id"]
        with tf.GradientTape() as tape:
            pos_score = self._net(sequence_ids, target_id)
            neg_score = self._net(sequence_ids, neg_id)
            loss = self._loss_func(pos_score, neg_score)
            #print(loss)
        gradients = tape.gradient(loss, self._net.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._net.trainable_variables))
        
        return loss

    def _predict_batch_hinge(self, batch_data):
        """
        """
        prediction = self._net(batch_data["sequence"], training=False)
        return prediction
    
    def predict(self, input):
        """
        """
        pass
