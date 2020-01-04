"""
gru4rec system
Session-based Recommendations with Recurrent Neural Networks
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Input, Dense, Dropout, GRU, Embedding

from tf_impl_reco.utils.rec15_data import *


def create_model(args):   
    emb_size = 50
    hidden_units = 100
    size = emb_size

    inputs = Input(batch_shape=(args.batch_size, 1, args.train_n_items))
    gru, gru_states = GRU(hidden_units, stateful=True, return_state=True)(inputs)
    drop2 = Dropout(0.25)(gru)
    predictions = Dense(args.train_n_items, activation='softmax')(drop2)
    model = Model(inputs=inputs, outputs=[predictions])
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, \
        epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=categorical_crossentropy, optimizer=opt)
    model.summary()

    filepath='./model_checkpoint.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')
    callbacks_list = []
    return model


def get_states(model):
    print(model.state_updates)
    return [K.get_value(s) for s,_ in model.state_updates]


def get_metrics(model, args, train_generator_map, recall_k=20, mrr_k=20):

    test_dataset = SessionDataset(args.test_data, itemmap=train_generator_map)
    test_generator = SessionDataLoader(test_dataset, batch_size=args.batch_size)

    n = 0
    rec_sum = 0
    mrr_sum = 0

    with tqdm(total=args.test_samples_qty) as pbar:
        for feat, label, mask in test_generator:

            target_oh = to_categorical(label, num_classes=args.train_n_items)
            input_oh  = to_categorical(feat,  num_classes=args.train_n_items) 
            input_oh = np.expand_dims(input_oh, axis=1)
            
            pred = model.predict(input_oh, batch_size=args.batch_size)

            for row_idx in range(feat.shape[0]):
                pred_row = pred[row_idx] 
                label_row = target_oh[row_idx]

                rec_idx =  pred_row.argsort()[-recall_k:][::-1]
                mrr_idx =  pred_row.argsort()[-mrr_k:][::-1]
                tru_idx = label_row.argsort()[-1:][::-1]

                n += 1

                if tru_idx[0] in rec_idx:
                    rec_sum += 1

                if tru_idx[0] in mrr_idx:
                    mrr_sum += 1/int((np.where(mrr_idx == tru_idx[0])[0]+1))
            
            pbar.set_description("Evaluating model")
            pbar.update(test_generator.done_sessions_counter)

    recall = rec_sum/n
    mrr = mrr_sum/n
    return (recall, recall_k), (mrr, mrr_k)


def train_model(model, args, save_weights = False):
    train_dataset = SessionDataset(args.train_data)
    model_to_train = model
    batch_size = args.batch_size

    for epoch in range(1, 10):
        with tqdm(total=args.train_samples_qty) as pbar:
            loader = SessionDataLoader(train_dataset, batch_size=batch_size)
            for feat, target, mask in loader:
                
                real_mask = np.ones((batch_size, 1))
                for elt in mask:
                    real_mask[elt, :] = 0
                #hidden_states = get_states(model_to_train)[0]
                hidden_states = model_to_train.layers[1].states[0]
                hidden_states = tf.math.multiply(real_mask, hidden_states)
                #hidden_states = np.array(hidden_states, dtype=np.float32)
                model_to_train.layers[1].reset_states(hidden_states)

                input_oh = to_categorical(feat, num_classes=loader.n_items) 
                input_oh = np.expand_dims(input_oh, axis=1)

                target_oh = to_categorical(target, num_classes=loader.n_items)

                tr_loss = model_to_train.train_on_batch(input_oh, target_oh)

                pbar.set_description("Epoch {0}. Loss: {1:.5f}".format(epoch, tr_loss))
                pbar.update(loader.done_sessions_counter)
            
        if save_weights:
            print("Saving weights...")
            model_to_train.save('./GRU4REC_{}.h5'.format(epoch))
        
        (rec, rec_k), (mrr, mrr_k) = get_metrics(model_to_train, args, train_dataset.itemmap)

        print("\t - Recall@{} epoch {}: {:5f}".format(rec_k, epoch, rec))
        print("\t - MRR@{}    epoch {}: {:5f}".format(mrr_k, epoch, mrr))
        print("\n")

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Keras GRU4REC: session-based recommendations')
    parser.add_argument('--resume', type=str, help='stored model path to continue training')
    parser.add_argument('--train-path', type=str, \
        default='/home/lyt/workspace/recsys/data/rec15/rsc15_train_tr.txt')
    parser.add_argument('--dev-path', type=str, \
        default='/home/lyt/workspace/recsys/data/rec15/rsc15_train_valid.txt')
    parser.add_argument('--test-path', type=str, \
        default='/home/lyt/workspace/recsys/data/rec15/rsc15_test.txt')
    parser.add_argument('--batch-size', type=str, default=512)
    args = parser.parse_args()


    args.train_data = pd.read_csv(args.train_path, sep='\t', dtype={'ItemId': np.int64})
    args.dev_data   = pd.read_csv(args.dev_path,   sep='\t', dtype={'ItemId': np.int64})
    args.test_data  = pd.read_csv(args.test_path,  sep='\t', dtype={'ItemId': np.int64})
    
    args.train_n_items = len(args.train_data['ItemId'].unique()) + 1

    args.train_samples_qty = len(args.train_data['SessionId'].unique()) + 1
    args.test_samples_qty = len(args.test_data['SessionId'].unique()) + 1

    if args.resume:
        try:
            model = keras.models.load_model(args.resume)
            print("Model checkpoint '{}' loaded!".format(args.resume))
        except OSError:
            print("Model checkpoint could not be loaded. Training from scratch...")
            model = create_model(args)
    else:
        model = create_model(args)
    
    train_model(model, args, save_weights=True)
