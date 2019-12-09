import tensorflow as tf
from tensorflow import keras
import os

# import wandb
# from wandb.keras import WandbCallback

# wandb.init(project="custard")

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


def build_1D_branch(sequence_input):

    model = keras.layers.Conv1D(
        filters=12, kernel_size=(6), padding="same", data_format="channels_last"
    )(sequence_input)

    # x = Bidirectional(LSTM(16, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', dropout=tmp_dropout, recurrent_dropout=tmp_dropout))(x)
    # x = Bidirectional(LSTM(8, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', dropout=tmp_dropout, recurrent_dropout=tmp_dropout))(x)
    layer_out = keras.layers.Flatten()(model)

    return layer_out


def build_2D_branch(sequence_input):
    """
    fun creates the network architecture 
    necessary for the training and 
    evaluation.

    network architecture was empirical
    defined and hard coded.

    parameters:

    """

    model = keras.layers.Conv2D(
        filters=12, kernel_size=(6, 6), padding="same", data_format="channels_last"
    )(sequence_input)

    # model.add(keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(keras.layers.Dropout(0.2))

    # model.add(keras.layers.Conv2D(filters=128, kernel_size=(6, 6), padding="same",))
    # model.add(keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(keras.layers.Dropout(0.2))

    # model.add(keras.layers.Conv2D(filters=128, kernel_size=(6, 6), padding="same"))
    # model.add(keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(keras.layers.Dropout(0.2))

    # model.add(keras.layers.Flatten())
    model = keras.layers.Flatten()(model)
    return model


def build_multi_braches(dim_1=50, dim_2=20, dim_3=200):
    """
    create multi branches architecture
    """

    sequence_outputs = []
    sequence_inputs = []

    tensor_input = keras.layers.Input(shape=(dim_1, dim_2, 1))
    sequence_inputs.append(tensor_input)
    sequence_outputs.append(build_2D_branch(tensor_input))

    tensor_input = keras.layers.Input(shape=(dim_1, 4))
    sequence_inputs.append(tensor_input)
    sequence_outputs.append(build_1D_branch(tensor_input))

    tensor_input = keras.layers.Input(shape=(dim_2, 4))
    sequence_inputs.append(tensor_input)
    sequence_outputs.append(build_1D_branch(tensor_input))

    tensor_input = keras.layers.Input(shape=(dim_1, dim_3, 1))
    sequence_inputs.append(tensor_input)
    sequence_outputs.append(build_2D_branch(tensor_input))

    return sequence_inputs, sequence_outputs


def build_architecture(classes=2, dim_1=50, dim_2=20, branches=2):
    """
    create single branch model with 2D dotmatrix.
    """

    sequence_inputs, sequence_outputs = build_multi_braches(dim_1, dim_2)

    concatenated = keras.layers.concatenate(sequence_outputs)

    model = keras.layers.Dense(512)(concatenated)
    model = keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(model)
    model = keras.layers.BatchNormalization()(model)
    model = keras.layers.Dropout(0.2)(model)

    model = keras.layers.Dense(classes)(model)
    model = keras.layers.Softmax(axis=-1)(model)

    classification_model = keras.Model(sequence_inputs, model)
    return classification_model


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
        X_train,
        y_train,
        sample_weight=None,
        class_weight=None,
        reset_metrics=False,
        callbacks=[WandbCallback()],
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
