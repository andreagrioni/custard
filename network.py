import tensorflow as tf
from tensorflow import keras
import os

"""
this module define and create the
network model.
"""


def build_network(classes=2, dim_1=50, dim_2=20):
    """
    fun is a pipeline of steps that 
    build a trainable network.
    """
    model = build_architecture(classes, dim_1, dim_2)

    model = compile_network(model=model, optimizer=optimizer())
    return model


def build_architecture(classes=2, dim_1=50, dim_2=20):
    """
    fun creates the network architecture 
    necessary for the training and 
    evaluation.

    network architecture was empirical
    defined and hard coded.

    parameters:
    classes=number of classes
    """
    model = keras.models.Sequential()

    array_shape = (dim_1, dim_2, 1)

    model.add(
        keras.layers.Conv2D(
            filters=128,
            kernel_size=(6, 6),
            padding="same",
            data_format="channels_last",
            input_shape=array_shape,
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.LeakyReLU(alpha=0.3))
    # model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Conv2D(filters=128, kernel_size=(6, 6), padding="same",))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.LeakyReLU(alpha=0.3))
    # model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(6, 6), padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.LeakyReLU(alpha=0.3))
    # model.add(keras.layers.Dropout(0.2))

    # model.add(
    #     keras.layers.Flatten()
    # )

    #    need further investigation

    model.add(keras.layers.GlobalAveragePooling2D())

    model.add(keras.layers.Dense(512))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.3))
    #    model.add(keras.layers.Dropout(0.2))

    # model.add(keras.layers.Dense(300))
    # model.add(keras.layers.BatchNormalization())

    # model.add(keras.layers.LeakyReLU(alpha=0.3))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(classes))
    model.add(keras.layers.Softmax(axis=-1))
    return model


def optimizer():
    """
    fun defines the optimizer for the 
    network ( only adam ).
    """
    adam = keras.optimizers.Adam(
        learning_rate=1e-03, beta_1=0.9, beta_2=0.999, amsgrad=False
    )
    return adam


def compile_network(model, optimizer):
    """
    fun compile the network into a 
    trainable model
    """
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_on_batch_network(
    model, X_train, y_train,
):

    history = model.train_on_batch(
        X_train, y_train, sample_weight=None, class_weight=None, reset_metrics=False
    )

    return history


def predict_on_batch_network(model, X_test):
    """
    fun predict on input minibatch

    paramenters:
    model=input model
    X_test=db to predict
    """
    predictions = model.predict_on_batch(X_test)
    return predictions


def test_on_batch_network(
    model, X_test, y_test,
):

    history = model.test_on_batch(
        X_test, y_test, sample_weight=None, reset_metrics=False
    )

    return history


def load_model_network(model_file_path):
    """
    load h5 model.
    
    paramenters:
    path=dir path of model
    name=model file name
    """
    print(model_file_path)
    model = keras.models.load_model(model_file_path)
    return model


def save_model(model, path, name="my_model.h5"):
    model_file_path = os.path.join(path, name)
    model.save(model_file_path)
    del model
    return model_file_path


if __name__ == "__main__":
    pass
