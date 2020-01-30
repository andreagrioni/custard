from tensorflow import keras
import pandas as pd
import tensorflow as tf
import os
import network
import math
import tempfile
import misc
import datetime

# not able to install wandb
# import wandb
# from wandb.keras import WandbCallback

# wandb.init(project="custard")


"""
train network
"""


def network_callbacks(log_name, dest_path):
    """
    define callbacks funtions
    to run during network
    training

    log_name=lof file name
    dest_path=file destination path
    """
    log_path = os.path.join(dest_path, log_name)

    Early_stop = keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=0.1,
        patience=5,
        verbose=1,
        mode="auto",
        baseline=0.8,
        restore_best_weights=False,
    )

    csv_logger = keras.callbacks.CSVLogger(log_path, separator="\t", append=False)
    return [csv_logger, Early_stop]


def train_network(
    model, train_set, val_dataset, batch_size, epochs, model_name
):
    """
    fun train network on user 
    specified datasets

    parameters:
    model=trainable network
    train_set=X,y train tuple
    batch_size=batch size
    """

    # callbacks
    log_dir="logs/fit/" + model_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.05,
        patience=0,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=False)


    tmpdirname = "train_tmp"
    os.makedirs(tmpdirname, exist_ok=True)

    X_train, y_train = train_set

    print("validation dataset provided by user")
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tensorboard_callback, early_stop],
        #        callbacks=[WandbCallback()],
        validation_data=val_dataset,
        use_multiprocessing=True,
        verbose=1,
    )
    return model


def create_wd(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    return None

def do_training(train_opt, dataset):

    # train settings
    batch_size = train_opt["batch_size"]
    classes = train_opt["classes"]
    tensor_dim = train_opt["tensor_dim"]
    epochs = train_opt["epochs"]
    model_output_dir = train_opt["model_output_dir"]
    model_name = train_opt["model_name"]
    # creates wd
    create_wd(model_output_dir)

    # generate network
    model = network.build_network(classes=classes, shape=tensor_dim)
    # train network
    train_dataset = (dataset[0], dataset[1])
    val_dataset = (dataset[2], dataset[3])

    model = train_network(model, train_dataset, val_dataset, batch_size=batch_size, epochs=epochs, model_name=model_name)
    # save model
    return network.save_model(model, model_output_dir, name=model_name)


if __name__ == "__main__":
    pass
