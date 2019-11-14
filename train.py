from tensorflow import keras
import pandas as pd
import tensorflow as tf
import os
import network
import math
import tempfile
import misc

'''
train network
'''


# def network_callbacks(
#     log_name,
#     dest_path
#     ):
#     '''
#     define callbacks funtions
#     to run during network
#     training

#     log_name=lof file name
#     dest_path=file destination path
#     '''
#     log_path = os.path.join(
#         dest_path,
#         log_name
#     )

#     Early_stop = keras.callbacks.EarlyStopping(
#         monitor='val_accuracy',
#         min_delta=0,
#         patience=5,
#         verbose=0,
#         mode='auto',
#         baseline=None,
#         restore_best_weights=False
#     )

    # csv_logger = keras.callbacks.CSVLogger(
    #     log_path,
    #     separator="\t",
    #     append=False
    # )

#    return [Early_stop]


def run_epochs(
    model,
    train_set,
    batch_size,
    iteration,
    batches_limit,
    tmp_path
    ):
    '''
    fun controls one cycle training
    
    paramenters:
    model=trainable network
    training_set=train set as ohe array
    batc_size=samples per minibatches
    iteration=iteration number
    batches_limit=limit train to N batches
    tmp_path=temporary directory
    '''
    csv_log = os.path.join(
        tmp_path, f"{iteration}_train.csv"
    )
    f = open(csv_log, 'w')
    header = '\t'.join([
        "iteration",
        "batch",
        "train_batch_size",
        "train_loss",
        "train_acc",
        "val_batch_size",
        "val_loss",
        "val_acc"
        ])
    f.write(header + "\n")
    
    for batch, batch_data in enumerate(
        train_set, start=1
        ):

        X_train_ohe, X_test_ohe, y_train_dummies, y_test_dummies = batch_data

        history = network.train_on_batch_network(
            model,
            X_train_ohe,
            y_train_dummies,
        )

        train_batch_history = history

        history = network.test_on_batch_network(
            model,
            X_test_ohe,
            y_test_dummies,
        )

        test_batch_history = history
        
        log_csv = misc.print_history(
            iteration,
            batch,
            len(train_set),
            batches_limit,
            X_train_ohe.shape[0],
            X_test_ohe.shape[0],
            train_batch_history,
            test_batch_history
        )
        f.write(log_csv + '\n')

        if batch >= batches_limit:
            break
    f.close()
    return None


def run_iterations(
    model,
    train_set,
    batch_size,
    iterations,
    batches_limit,
    tmp_name
    ):
    '''
    fun controls iterative training

    paramenters:
    model=trainable network
    train_set=whole train set as array
    iteration=number of iterations
    batches_limit=train each iteration 
    on N batches.
    '''

    for iteration in range(
        0, iterations + 1
    ):
        print()
        print("iteration:",
        iteration, "of total:",
        iterations, "batches_limit",batches_limit, sep="\t")

        run_epochs(
            model=model,
            train_set=train_set,
            batch_size=batch_size,
            iteration=iteration,
            batches_limit=batches_limit,
            tmp_path=tmp_name,
        )

    return model


def train_network(
    model,
    train_set,
    batch_size=32,
    batches_limit=None,
    iterations=None
    ):
    '''
    fun train network on user 
    specified datasets

    paramenters:
    model=trainable network
    train_set=ImageBatchGenerator
    batch_size=batch size
    batches_limit=define a max number 
    of batches to use for training
    iterations=max number of iterations
    '''
    tmpdirname="train_tmp"
    os.makedirs(
        tmpdirname, exist_ok=True)

    if not iterations:
        iterations = math.ceil(
            len(train_set)/batch_size
        )
    if not batches_limit:
        batches_limit = len(train_set)
    # RUN ITERATIONS MODULE
    model = run_iterations(
            model=model,
            train_set=train_set,
            batch_size=batch_size,
            iterations=iterations,
            batches_limit=batches_limit,
            tmp_name=tmpdirname
    )
    return model

if __name__ == "__main__":
    pass