"""
reimplement Matrix Factorization Techniques for Recommender Systems (Yahoo 2009)
"""
import sys
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

tf.enable_eager_execution()


class MF_Netflix(layers.Layer):
    def __init__(self, user_count, item_count, hidden_dim):
        super(MF_Netflix, self).__init__()
        regularizer = keras.regularizers.l2(0.0)
        self.user_layer = layers.Embedding(user_count, hidden_dim, \
            name="user_emb_layer", embeddings_regularizer=regularizer)
        self.item_layer = layers.Embedding(item_count, hidden_dim, \
            name="item_emb_layer", embeddings_regularizer=regularizer)

    def call(self, inputs):
        """
        inputs:
            batch_user_ids
            batch_item_ids
        outputs:
            the prediction, user-item score
        """
        batch_user_ids, batch_item_ids = inputs
        batch_user_emb = self.user_layer(batch_user_ids)
        batch_item_emb = self.item_layer(batch_item_ids)
        batch_score = tf.reduce_sum(\
            tf.multiply(batch_user_emb, batch_item_emb), \
                axis=1)
        return batch_score


def train_loop(args, train_dataset, dev_dataset, test_dataset=None):
    """
    the train loop function
    """
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        # build model
        user_ids = keras.Input(shape=(1,), dtype=tf.int32, name="user_id")
        item_ids = keras.Input(shape=(1,), dtype=tf.int32, name="item_id")
        batch_score = MF_Netflix(args.user_count, args.item_count, args.hidden_dim)(\
            [user_ids, item_ids])
        model = keras.Model(inputs={"user_id":user_ids, "item_id":item_ids}, \
            outputs=batch_score)
        # build the model train setting
        optimizer = keras.optimizers.Adam(args.learning_rate)
        loss = keras.losses.MeanSquaredError()
        metrics = [keras.metrics.MeanSquaredError()]
        model.compile(optimizer, loss=loss, metrics=metrics)
    # make the training loop and evaluation
    checkpoint_callback = keras.callbacks.ModelCheckpoint(\
        filepath=args.model_path, save_best_only=True)
    tensorbaord_callback = keras.callbacks.TensorBoard(log_dir=args.summary_dir)
    steps_per_epoch = 62500
    model.fit(train_dataset, epochs=args.epochs, \
        callbacks=[checkpoint_callback, tensorbaord_callback], \
            validation_data=dev_dataset, steps_per_epoch=steps_per_epoch)


if __name__ == "__main__":
    user_count = 100
    item_count = 10000
    hidden_dim = 300
    batch_user_ids = np.random.randint(1, user_count, 32)
    batch_item_ids = np.random.randint(1, item_count, 32)

    mf_obj = MF_Netflix(user_count, item_count, hidden_dim)
    
    print(mf_obj([batch_user_ids, batch_item_ids]))