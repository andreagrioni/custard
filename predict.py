import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import pre_processing
import os
import network


def model_predict(
    model, dataset, batch_size=32
    ):
    '''
    predict dataset with selected
    model.

    paramenters:
    model=model object
    dataset=one hot encoding array
    batch_size=dataset batch size
    '''
    for batch, batch_data in enumerate(
        dataset, start=1):
        X_test, y_test = batch_data
        
        predictions = network.predict_on_batch_network(
            model, X_test
        )
        print(predictions)
    return predictions


if __name__ == "__main__":
    path = '/home/angri/Desktop/projects/custard_testing/my_model.h5'
    target_tsv = "/home/angri/Desktop/projects/custard_testing/custard_toy.tsv"
    model = network.load_model_network(path)
    dataset = pre_processing.load_dataset(
        target_tsv, scope='evaluate'
        )
    predictions = model_predict(
    model, dataset
    )