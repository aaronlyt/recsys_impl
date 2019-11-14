"""
This module prepares and runs the whole system.
"""
import sys
sys.path.append('../../')
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle
import argparse
import logging

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

from tf_impl_reco.utils.movielens import read_movielens_20M
from tf_impl_reco.utils.netflix import make_netflix_dataset
from tf_impl_reco.component.mf_movielen import train_loop


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--learning_rate', type=float, default=1e-2)
    train_settings.add_argument('--batch_size', type=int, default=32)
    train_settings.add_argument('--epochs', type=int, default=10)
    train_settings.add_argument('--user_count', type=int, default=0)
    train_settings.add_argument('--item_count', type=int, default=0)

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--hidden_dim', type=int, default=100)
 
    
    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', type=str, \
        default="/home/lyt/workspace/recsys/data/netflixprize/training_set")
    path_settings.add_argument('--dev_files', type=str, \
        default="/home/lyt/workspace/recsys/data/netflixprize/probe.txt")
    path_settings.add_argument('--test_files', type=str, \
        default="/home/lyt/workspace/recsys/data/netflixprize/qualifying.txt")
    path_settings.add_argument('--data_dir', type=str, \
        default="/home/lyt/workspace/recsys/tf_impl_reco/data/")
    path_settings.add_argument('--model_path', default='../data/models/movielen_mf_md', \
        help='the dir to store models')
    path_settings.add_argument('--summary_dir', default='../data/summary/movielen_mf/', \
        help='the dir to write tensorboard summary')
    return parser.parse_args()


def train(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("movielen mf")
    train_dataset, dev_data, test_data, movie_count, user_count = \
        make_netflix_dataset(args.train_files, args.dev_files, args.test_files, \
            args.data_dir, args.batch_size, args.epochs)
    args.user_count = user_count
    args.item_count = movie_count

    logger.info('Initialize the model...')
    model = train_loop(args, train_dataset, dev_data)
    logger.info('Done with model training!')

def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    if args.train:
        train(args)

if __name__ == '__main__':
    run()