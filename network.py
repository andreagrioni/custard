import tensorflow as tf
from tensorflow import keras

'''
this module define and create the
network model.
'''


def build_network():
    '''
    fun is a pipeline of steps that 
    build a trainable network.
    '''
    model = build_architecture(
        NETWORK_ARCHITECTURE
    )
    model = compile_network(
        model=model,
        optimizer=optimizer()
    )
    return model


def build_architecture(
):
    '''
    fun creates the network architecture 
    necessary for the training and 
    evaluation.

    network architecture was empirical
    defined and hard coded.

    paramenters:
    NETWORK_ARCHITECTURE=list of tuples with layer and filters/neurons
    '''
    model = keras.models.Sequential()

    model.add(
        keras.layers.Conv2D(
            filters=128,
            kernel_size=(6, 6),
            padding='same',
            data_format="channels_last",
            input_shape=(50, 20, 3)
        )
    )
    model.add(
        keras.layers.LeakyReLU(
            alpha=0.3
        )
    )
    model.add(
        keras.layers.BatchNormalization())
    model.add(
        keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        )
    )
    model.add(
        keras.layers.Dropout(0.2)
    )
    model.add(
        keras.layers.Conv2D(
            filters=64,
            kernel_size=(6, 6),
            padding='same',
        )
    )
    model.add(
        keras.layers.LeakyReLU(
            alpha=0.3
        )
    )
    model.add(
        keras.layers.BatchNormalization())
    model.add(
        keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        )
    )
    model.add(
        keras.layers.Dropout(0.2)
    )

    model.add(
        keras.layers.Conv2D(
            filters=32,
            kernel_size=(6, 6),
            padding='same'
        )
    )
    model.add(
        keras.layers.LeakyReLU(
            alpha=0.3
        )
    )
    model.add(
        keras.layers.BatchNormalization())
    model.add(
        keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        )
    )
    model.add(
        keras.layers.Dropout(0.2)
    )

    model.add(
        keras.layers.Flatten()
    )

    model.add(
        keras.layers.Dense(512)
    )

    model.add(
        keras.layers.LeakyReLU(
            alpha=0.3
        )
    )
    model.add(
        keras.layers.BatchNormalization())
    model.add(
        keras.layers.Dropout(0.2)
    )

    model.add(
        keras.layers.Dense(300)
    )

    model.add(
        keras.layers.LeakyReLU(
            alpha=0.3)
    )
    model.add(
        keras.layers.BatchNormalization())
    model.add(
        keras.layers.Dropout(0.2)
    )

    model.add(
        keras.layers.Dense(2)
    )
    model.add(
        keras.layers.LeakyReLU(
            alpha=0.3
        )
    )
    model.add(
        keras.layers.Softmax(
            axis=-1
        )
    )
    return model


def optimizer():
    '''
    fun defines the optimizer for the 
    network ( only adam ).
    '''
    adam = keras.optimizers.Adam(
        learning_rate=1e-03,
        beta_1=0.9,
        beta_2=0.999,
        amsgrad=False
    )
    return adam


def compile_network(
    model=model,
    optimizer=optimizer
):
    '''
    fun compline the network into a 
    trainable model
    '''
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def fit_network(
    X_train,
    y_train,
    batch_size,
    callbacks,

):
    '''
    fun fits the model with 
    the training set.

    paramenters:
    X_train=train set
    y_train=label train set
    batch_size=int
    callbacks=Keras callbacks
    '''
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=32,
        epochs=1,
        verbose=1,
        callbacks=callbacks(log_name, tmp_path),
        # validation_split=0.2,
        #validation_data=(X_val, y_val),
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        # validation_steps=None,
        # validation_freq=[2,5,10],
        max_queue_size=10,
        workers=0,
        use_multiprocessing=True
    )
    return history
