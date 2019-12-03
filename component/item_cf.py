"""
item based callaborate filterinig algorithm
Amazon.com Recommendations Item-to-Item Collaborative Filtering
"""

import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mlpc


def offline_cal_smi(train_path):
    """
    @train_path: csv file
    read train file and calculate the item similarity
    """
    dataset = pd.read_csv(train_path, chunksize=1e7)
    
    pass


def validation_dev(dev_path):
    """
    predict and evaluation
    """
    pass



if __name__ == "__main__":
    pass