import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import table2ohe
import os
import network
import numpy as np


def model_predict(model, dataset, batch_size=32):
    """
    predict dataset with selected
    model.

    paramenters:
    model=model object
    dataset=one hot encoding array
    batch_size=dataset batch size
    """
    X_pred = list()
    y_pred = list()
    for batch, batch_data in enumerate(dataset, start=1):
        X_test, y_test = batch_data

        prediction = network.predict_on_batch_network(model, X_test)
        X_pred.append(prediction)
        y_pred.append(y_test)

    X_pred = np.concatenate(X_pred)
    y_pred = np.concatenate(y_pred)
    return (X_pred, y_pred)


if __name__ == "__main__":
    path = "/home/angri/Desktop/projects/custard_testing/my_model.h5"
    target_tsv = "/home/angri/Desktop/projects/custard_testing/custard_toy.tsv"
    model = network.load_model_network(path)
    dataset = pre_processing.load_dataset(target_tsv, scope="evaluate")
    predictions = model_predict(model, dataset)

