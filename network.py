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


def encoder_sequence_branch(sequence_input):

    x = Conv1D(filters=128, kernel_size=6)(sequence_input)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, padding="same")(x)
    x = Dropout(rate=0.5, noise_shape=None, seed=None)(x)

    x = Conv1D(filters=tmp_filter_num, kernel_size=15, strides=1, padding="same")(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, padding="same")(x)
    x = Dropout(rate=0.5, noise_shape=None, seed=None)(x)

    x = Conv1D(
        filters=int(tmp_filter_num / 2), kernel_size=10, strides=1, padding="same"
    )(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, padding="same")(x)
    x = Dropout(rate=0.5, noise_shape=None, seed=None)(x)

    # x = Bidirectional(LSTM(16, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', dropout=tmp_dropout, recurrent_dropout=tmp_dropout))(x)
    # x = Bidirectional(LSTM(8, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', dropout=tmp_dropout, recurrent_dropout=tmp_dropout))(x)
    conservation_output = Flatten()(x)

    return conservation_output


def build_2D_branch(sequence_input=None, classes=2, dim_1=50, dim_2=20):
    """
    fun creates the network architecture 
    necessary for the training and 
    evaluation.

    network architecture was empirical
    defined and hard coded.

    parameters:
    classes=number of classes
    """
    # model = keras.models.Sequential()

    # array_shape = (dim_1, dim_2, 1)

    model = keras.layers.Conv2D(
        filters=128, kernel_size=(6, 6), padding="same", data_format="channels_last"
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


def build_multi_braches(classes=2, dim_1=50, dim_2=20):
    """
    create multi branches architecture
    """

    sequence_outputs = []
    sequence_inputs = []

    for i in range(0, 3):

        sequence_input = keras.layers.Input(shape=(dim_1, dim_2, 1))
        sequence_outputs.append(
            build_2D_branch(
                sequence_input=sequence_input, classes=2, dim_1=50, dim_2=20
            )
        )

    return sequence_inputs, sequence_outputs


def build_architecture(classes=2, dim_1=50, dim_2=20, branches=2):
    """
    create single branch model with 2D dotmatrix.
    """

    sequence_inputs, sequence_outputs = build_multi_braches(
        classes=2, dim_1=50, dim_2=20
    )

    model = keras.layers.concatenate(sequence_outputs)

    model.add(keras.layers.Dense(512))
    model.add(keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(classes))
    model.add(keras.layers.Softmax(axis=-1))

    classification_model = Model(sequence_inputs, model)
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
