from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import shutil
#import IPython.display as display
#import PIL
import numpy as np
#import matplotlib.pyplot as plt
import network
import train
import pre_processing


def do_training(target_file):
    # make dataset from custard table
    # as tuple of value, labels.
    train_set = pre_processing.load_dataset(
        target_tsv=target_file
    )

    # generate network
    model = network.build_network()


    # train network
    history = train.train_network(
        model,
        train_set,
        batches_limit=None,
        iterations=None
    )


if __name__ == "__main__":
    target_file = 'pre_processing_test.tsv'
    do_training(target_file)