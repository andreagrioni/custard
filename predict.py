import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import pre_processing
import os

def load(path, name='my_model.h5'):
    model_file_path = os.path.join(path, name)
    model = load_model(model_file_path)
    return model


def model_predict(
    model, dataset, batch_size=32
    ):
    predictions = model.predict(
        dataset, batch_size=batch_size,
        verbose=0, steps=None,
        callbacks=None,
        max_queue_size=10, workers=1,
        use_multiprocessing=False
        )
    return predictions

if __name__ == "__main__":
    path = './'
    target_tsv = "pre_processing_test.tsv"
    model = load(path)
    df_ohe, df_labels = pre_processing.load_dataset(
        target_tsv,
        scope='evaluation'
        )
    
    predictions = model_predict(
    model, df_ohe, batch_size=32
    )