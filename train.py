from tensorflow import keras
import pandas as pd
import tensorflow as tf
import os
import network
import math
import tempfile
'''
train network
'''


def update_history(
    history,
    log_history,
    iteration=None
    ):
    '''
    update dataframe with training
    history.

    parameters:
    history=df with current history
    log_history=df with previous histories
    iteration=iteration number
    '''
    df_history = pd.DataFrame(
        history
        )
    if iteration:
        df_history['iteration'] = iteration
    updated_history = log_history.append(
        df_history
        )
    return updated_history


def network_callbacks(
    log_name,
    dest_path
    ):
    '''
    define callbacks funtions
    to run during network
    training

    log_name=lof file name
    dest_path=file destination path
    '''
    log_path = os.path.join(
        dest_path,
        log_name
    )

    Early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0,
        patience=5,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=False
    )

    csv_logger = keras.callbacks.CSVLogger(
        log_path,
        separator="\t",
        append=False
    )

    return [csv_logger, Early_stop]


def run_epochs(
    model,
    train_set,
    batch_size,
    iteration,
    batches_limit,
    tmp_path,
    log_history
    ):
    '''
    fun controls one cycle training
    
    paramenters:
    training_set=train set as ohe array
    batches_limit=limit train to N batches
    iteration=iteration number
    log_history=df of train history
    '''
    for batch, batch_data in enumerate(
        train_set
    ):
        X_train, y_train = batch_data
        
        log_name = f'{iteration}_{batch}_train.csv'

        callbacks = network_callbacks(
            log_name,
            tmp_path
        )

        history = network.fit_network(
            model,
            X_train,
            y_train,
            batch_size,
            callbacks
        )
        
        epoch_hist = history.history

        metrics = [f'{metric}|{value[0]:.2f}' if not 'loss' in metric else  f'{metric}|{value[0]:.2E}' for metric, value in epoch_hist.items()]

        print(
            "training\tbatch:", batch,
            "of total:",
            len(train_set),
            '\t'.join(metrics),
            sep = "\t"
        )

        # UPDATE HISTORY
        log_history = update_history(
                epoch_hist,
                log_history,
                iteration
                )

        if batch >= batches_limit:
            break
    return log_history


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
    log_history = pd.DataFrame()
    for iteration in range(
        0, iterations
    ):
        print("iteration:",
        iteration, "of total:",
        iterations, sep="\t")
        print("\tbatch limit:",
        batches_limit, sep="\t")

        history = run_epochs(
            model=model,
            train_set=train_set,
            batch_size=batch_size,
            iteration=iteration,
            batches_limit=batches_limit,
            tmp_path=tmp_name,
            log_history=log_history
        )
        log_history = update_history(
                history,
                log_history
                )
    return log_history, model


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
    history, model = run_iterations(
            model=model,
            train_set=train_set,
            batch_size=batch_size,
            iterations=iterations,
            batches_limit=batches_limit,
            tmp_name=tmpdirname
    )
    return history, model

if __name__ == "__main__":
    pass