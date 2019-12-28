# purpose
learning

# environment
* python 3.6
* tensorflow 2.0(keras API)
* tqdm

# recsys algorithms

## matrix factorization

### reference algorithm
matrix factorization with temporal dynamics and bias(MF_Alg: Matrix Factorization Techniques for Recommender Systems)

### training script
* training dataset
netflix dataset
* dataset processing
utils/netflix.py, two kinds of dataset api: tfrecord and tensorarray; 

big dataframe multiprocessing process

* train script: train/run_basic_exp.py

### evaluation result
MF_Alg: 49ms/step - loss: 0.8175 - mean_squared_error: 0.7460 - val_loss: 1.0178 - val_mean_squared_error: 0.9235

## sequence recommendation system(for learning purpose)

### algorithms
DNN, LSTM, LSTM with attention

### training script

* dataset
movielens 20M
sequence processing: utils/seq_process.py

* train script: train/run_seq_exp.py

* custom keras metrics
NCG
MRR

* evaluation
dnn loss: tf.Tensor(0.26367342, shape=(), dtype=float32) ---epoch: 0, mrr:0.0223
LSTM: loss: tf.Tensor(0.13960448, shape=(), dtype=float32) ---epoch: 3, mrr:0.0433
attention: test OOM
### reference
spotlight https://github.com/maciejkula/spotlight

## TODO
GRU4Rec
