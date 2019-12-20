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


def build_network(classes=2, shapes=(50,20,200)):
    """
    fun is a pipeline of steps that 
    build a trainable network.
    """
    dim_1, dim_2, dim_3 = *shapes
    model = build_architecture(classes, dim_1, dim_2, dim_3)

    model = compile_network(model=model, optimizer=optimizer())
    return model


def build_1D_branch(sequence_input):
    """
    this branch is a simple branch
    of 1 Conv1D and MaxPooling2D.

    """
    branch = keras.layers.Conv1D(
        filters=12, kernel_size=(6), padding="same", data_format="channels_last"
    )(sequence_input)
    branch = keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(branch)
    branch = keras.layers.MaxPooling1D(pool_size=(2, 2))(branch)
    branch = keras.layers.Flatten()(branch)

    return branch


def build_2D_branch(sequence_input):
    """
    this branch is a simple branch
    of 1 Conv2D and MaxPooling2D.

    """

    branch = keras.layers.Conv2D(
        filters=12, kernel_size=(6, 6), padding="same", data_format="channels_last"
    )(sequence_input)
    branch = keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(branch)
    branch = keras.layers.MaxPooling2D(pool_size=(2, 2))(branch)
    branch = keras.layers.Flatten()(branch)
    
    return branch


def build_multi_braches(dim_1=50, dim_2=20, dim_3=200):
    """
    use Keras Model API to build NN model
    with 2 branches: 2D dot matrix (2D ConvNet)
    and binding site conservation (1D ConvNet)
    """

    # store NN inputs and outputs to lists
    sequence_outputs = []
    sequence_inputs = []

    # define first branch 2D ConvNet
    tensor_input = keras.layers.Input(shape=(dim_1, dim_2, 1))
    sequence_inputs.append(tensor_input)
    sequence_outputs.append(build_2D_branch(tensor_input))

    # define second branch 1D ConvNet
    tensor_input = keras.layers.Input(shape=(dim_3, 1))
    sequence_inputs.append(tensor_input)
    sequence_outputs.append(build_1D_branch(tensor_input))

    return sequence_inputs, sequence_outputs


def build_architecture(classes, dim_1, dim_2, dim_3):
    """
    function creates a NN of 2 branches (CNN)
    """

    sequence_inputs, sequence_outputs = build_multi_braches(dim_1, dim_2)

    concatenated = keras.layers.concatenate(sequence_outputs)

    model = keras.layers.Dense(512)(concatenated)
    model = keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(model)
    model = keras.layers.BatchNormalization()(model)
    model = keras.layers.Dropout(0.5)(model)

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
