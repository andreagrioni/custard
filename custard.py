from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import shutil
import numpy as np
import network
import train
import pre_processing


def data_gen(target_file):
    train_set = pre_processing.load_dataset(
        target_tsv=target_file
    )
    return train_set

def do_training(dataset):
    # generate network
    model = network.build_network()

    # train network
    history, model = train.train_network(
        model,
        dataset,
        batches_limit=None,
        iterations=None
    )
    return history, model


if __name__ == "__main__":
    evaluate = True
    train = False
    target_file = 'pre_processing_test.tsv'
    dataset = data_gen(target_file)

    if train:
        history, model = do_training(dataset)
        network.save_model(model, os.getcwd())
    elif evaluate:
        print(os.getcwd())
        
