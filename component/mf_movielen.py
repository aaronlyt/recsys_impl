"""
reimplement Matrix Factorization Techniques for Recommender Systems (Yahoo 2009)
"""
import sys
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class MF_Netflix(layers.Layer):
    def __init__(self, user_count, item_count, hidden_dim, global_mean):
        super(MF_Netflix, self).__init__()
        self.item_count = item_count
        regularizer = keras.regularizers.l2(1e-6)
        self.user_layer = layers.Embedding(user_count, hidden_dim, \
            name="user_emb_layer", embeddings_regularizer=regularizer, \
                embeddings_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
        self.item_layer = layers.Embedding(item_count, hidden_dim, \
            name="item_emb_layer", embeddings_regularizer=regularizer, \
                embeddings_initializer=tf.keras.initializers.RandomNormal(0.0, 0.1))
        
        self.user_bias = layers.Embedding(user_count, 1, name="user_bias", \
            embeddings_regularizer=regularizer)
        self.item_bias = layers.Embedding(item_count, 1, name="item_bias", \
            embeddings_regularizer=regularizer)
        self.user_dev_param = layers.Embedding(user_count, 1, name="user_dev_param", \
            embeddings_regularizer=regularizer)
        self.item_bin_param = layers.Embedding(31, 1, name="item_bin_param", \
            embeddings_regularizer=regularizer)
        self.global_mean = global_mean
    """
    def build(self, inputs_shape):
        super(MF_Netflix, self).build(inputs_shape)
        self.item_bin_param = self.add_weight(shape=(self.item_count, 31), \
            trainable=True, name="item_bin_param", dtype=tf.float32, \
                )
    """
    def call(self, inputs):
        """
        inputs:
            batch_user_ids
            batch_movie_ids
        outputs:
            the prediction, user-item score
        """
        #(bssize,)
        batch_user_ids, batch_movie_ids, item_bin_ids, user_time_dev = inputs
        batch_user_emb = self.user_layer(batch_user_ids)
        batch_item_emb = self.item_layer(batch_movie_ids)
        
        batch_user_base_bias = tf.reshape(self.user_bias(batch_user_ids), [-1])
        batch_user_dev_bias = tf.reshape(self.user_dev_param(batch_user_ids), [-1])
        # (bssize )
        batch_user_bias = batch_user_base_bias + batch_user_dev_bias * user_time_dev
        #batch_user_bias = batch_user_base_bias

        batch_item_base_bias = tf.reshape(self.item_bias(batch_movie_ids), [-1])
        #item_bin_ids = tf.stack([batch_movie_ids, item_bin_ids], axis=1)
        #batch_item_bin_bias = tf.gather_nd(self.item_bin_param, item_bin_ids)
        batch_item_bin_bias = tf.reshape(self.item_bin_param(item_bin_ids), [-1])
        batch_item_bias = batch_item_base_bias + batch_item_bin_bias
        #batch_item_bias = batch_item_base_bias
        # (bssize)
        batch_score = tf.reduce_sum(\
            tf.multiply(batch_user_emb, batch_item_emb), \
                axis=1) + batch_user_bias + batch_item_bias + self.global_mean
        return batch_score


def train_loop(args, train_dataset, dev_dataset, global_mean=0.0, test_dataset=None):
    """
    the train loop function
    """
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        # build model
        user_ids = keras.Input(shape=(), dtype=tf.int32, name="user_id")
        movie_ids = keras.Input(shape=(), dtype=tf.int32, name="movie_id")
        item_bin_ids = keras.Input(shape=(), dtype=tf.int32, name="item_time_bias")
        user_time_dev = keras.Input(shape=(), dtype=tf.float32, name="user_time_dev")
        batch_score = MF_Netflix(args.user_count, args.item_count, args.hidden_dim, global_mean)(\
            [user_ids, movie_ids, item_bin_ids, user_time_dev])
        model = keras.Model(inputs={"user_id":user_ids, "movie_id":movie_ids, \
            "item_time_bias": item_bin_ids, "user_time_dev": user_time_dev}, \
            outputs=batch_score)
        # build the model train setting
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                args.learning_rate,
                decay_steps=20000,
                decay_rate=0.96,
                staircase=True)
        optimizer = keras.optimizers.Adam(args.learning_rate)
        #optimizer = keras.optimizers.RMSprop(args.learning_rate)
        #optimizer = keras.optimizers.SGD(args.learning_rate)
        loss = keras.losses.MeanSquaredError()
        metrics = [keras.metrics.MeanSquaredError()]
        model.compile(optimizer, loss=loss, metrics=metrics)
    # make the training loop and evaluation
    checkpoint_callback = keras.callbacks.ModelCheckpoint(\
        filepath=args.model_path, save_best_only=True, save_weights_only=True)
    tensorbaord_callback = keras.callbacks.TensorBoard(log_dir=args.summary_dir, \
        histogram_freq=1)
    steps_per_epoch = args.steps_per_epoch
    model.fit(train_dataset, epochs=args.epochs, \
        callbacks=[checkpoint_callback, tensorbaord_callback], \
            validation_data=dev_dataset, steps_per_epoch=steps_per_epoch, \
                validation_steps=args.val_steps)


if __name__ == "__main__":
    user_count = 100
    item_count = 10000
    hidden_dim = 300
    batch_user_ids = np.random.randint(1, user_count, 32)
    batch_movie_ids = np.random.randint(1, item_count, 32)

    mf_obj = MF_Netflix(user_count, item_count, hidden_dim)
    
    print(mf_obj([batch_user_ids, batch_movie_ids]))
