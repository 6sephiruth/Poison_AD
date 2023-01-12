import numpy as np
import tensorflow as tf

import os

from utils import *

SEED = 0
DATASET = 'mnist'

tf.random.set_seed(SEED)
np.random.seed(SEED)

# designate gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# enable memory growth
physical_devices = tf.config.list_physical_devices('GPU')
for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)
os.environ['TF_DETERMINISTIC_OPS'] = '0'

# dataset load
if DATASET == 'mnist':
    train, test = mnist_data()
# elif DATASET == 'cifar10':
#     train, test = cifar10_data()
#     # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


make_backdoor_dataset(train, 100)