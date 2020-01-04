import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import tensorflow as tf


class SimpleLstmModel(tf.keras.Model):
    """ Simple lstm model with two lstm """
    def __init__(self, units=10, stateful=True):
        super(SimpleLstmModel, self).__init__()
        self.lstm_0 = tf.keras.layers.LSTM(units=units, stateful=stateful, return_sequences=True)
        self.lstm_1 = tf.keras.layers.LSTM(units=units, stateful=stateful, return_sequences=True)

    def call(self, inputs):
        """
        :param inputs: [batch_size, seq_len, 1]
        :return: output tensor
        """
        x = self.lstm_0(inputs)
        x = self.lstm_1(x)
        return x

def main():
    model = SimpleLstmModel(units=1, stateful=True)
    x = tf.constant(np.random.sample([1, 1, 1]), dtype=tf.float32)
    output = model(x)
    print(model.state_updates)



if __name__ == "__main__":
    main()
