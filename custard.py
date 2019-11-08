from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import math
import shutil
import IPython.display as display
import PIL
import numpy as np
import matplotlib.pyplot as plt
import network
import train


# generate network
model = network.build_network()

# make dataset TODO

# train network
history = train.train_network(
    model,
    train_set,
    batch_size_train,
    batches_limit=None,
    iterations=None
)
