"""
This module prepares and runs the whole system.
"""
import sys
sys.path.append('../../')
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
import pickle
import argparse
import logging
import json
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
    train_settings.add_argument('--batch_size', type=int, default=64)
    train_settings.add_argument('--epochs', type=int, default=10)
    train_settings.add_argument('--user_count', type=int, default=0)
    train_settings.add_argument('--item_count', type=int, default=0)
    train_settings.add_argument('--steps_per_epoch', type=int, default=0)
    train_settings.add_argument('--val_steps', type=int, default=0)

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
    meta_path = os.path.join(args.data_dir, "datas.dump")
    data = json.load(open(meta_path, "r"))
    dev_len = data["dev_count"]
    train_lens = data["train_count"]
    buffer_size = args.batch_size * 50000
    logger = logging.getLogger("movielen mf")
    train_dataset, dev_data, test_data = \
        make_netflix_dataset(args.data_dir, buffer_size, args.batch_size, args.epochs)
    args.user_count = 480189
    args.item_count = 17770
    args.steps_per_epoch = math.ceil(train_lens / args.batch_size)
    args.val_steps = math.ceil(dev_len / args.batch_size)
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