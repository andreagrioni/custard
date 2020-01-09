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


def build_1D_branch(sequence_input):
    """
    this branch is a simple branch
    of 1 Conv1D and MaxPooling2D.

    parameters:
    sequence_input=tensor
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

    parameters:
    sequence_input=tensor
    """

    branch = keras.layers.Conv2D(
        filters=12, kernel_size=(6, 6), padding="same", data_format="channels_last"
    )(sequence_input)
    branch = keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(branch)
    branch = keras.layers.MaxPooling2D(pool_size=(2, 2))(branch)
    branch = keras.layers.Flatten()(branch)
    
    return branch


def build_multi_braches(shape):
    """
    use Keras Model API to build NN model
    with 2 branches: 2D dot matrix (2D ConvNet)
    and binding site conservation (1D ConvNet)

    parameters:
    shape=tensors shapes
    """

    # define conv_net 2d between binding sites and miRNA
    ## declare input tensor
    tensor_input = keras.layers.Input(shape=(shape[0], shape[1], 2))
    ## define 2D conv_net input
    conv_net_2d = build_2D_branch(tensor_input)

    return tensor_input, conv_net_2d


def add_ann(concatenated, classes):
    """
    build the ANN layers for the final
    input classification

    paramenters:
    concatenated=concatenations of previuous layers
    classes=number of predicted classes
    """
    # build ANN model layers
    model = keras.layers.Dense(512)(concatenated)
    model = keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(model)
    model = keras.layers.BatchNormalization()(model)
    model = keras.layers.Dropout(0.5)(model)

    model = keras.layers.Dense(classes)(model)
    model = keras.layers.Softmax(axis=-1)(model)

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

def build_network(classes, shape):
    """
    function creates a NN of 2 branches (CNN)
    """

    # generate CNN branches
    sequence_inputs, sequence_outputs = build_multi_braches(shape)

    # concatenate branches
#    concatenated = keras.layers.concatenate(sequence_outputs)

    # build ANN model layers
    model_architecture = add_ann(sequence_outputs, classes)
    
    # merge model
    classification_model = keras.Model(sequence_inputs, model_architecture)


    # compile model to trainable
    model = compile_network(model=classification_model, optimizer=optimizer())
    
    return model


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
