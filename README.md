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

## sequence recommendation system
### algorithms
DNN, LSTM, LSTM with attention

### training script

* dataset
movielens 20M
sequence processing: utils/seq_process.py

* train script: train/run_seq_exp.py

* custom keras metrics
NCG

#### working
sampling strategy: sampling with the frequency 
loss function: multiple negtive samples

optimizing: sequence loss, the loss function or the sampling strategy
