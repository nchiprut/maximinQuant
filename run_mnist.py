# %%
import tensorflow as tf
import numpy as np
import larq as lq
from pathlib import Path
import datetime
from tensorboard.plugins.hparams import api as hp

from misc import get_datasets, get_cnn, get_fc
from misc import TensorBoardExtra, MaybeSteSign
from quantizedModel import QuantModel


# %%
should_quantize = tf.Variable(True, trainable=False, dtype=tf.bool, name='shuold_quant')
dataset = 'mnist'
bs = 256

train_batches, test_batches, metadata = get_datasets(dataset, bs)
input_shape = metadata.features['image'].shape
num_classes = metadata.features['label'].num_classes
# %%
log_dir = "logs/mnist/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

n_epochs = 40
decay_epochs = 8
min_decay = 0.9
max_decay = 1.5
min_start_lr = 5e-3
max_start_lr = 2e-3
lamda = 1e3

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
tensorboard_extra = TensorBoardExtra(log_dir=log_dir)

# layers = get_cnn(input_shape, lambda: MaybeSteSign(to_quant=should_quantize), num_classes)
layers = get_fc(input_shape, lambda: MaybeSteSign(to_quant=should_quantize), num_classes)
quant_model = QuantModel(layers, should_quantize)

decay_steps = (metadata.splits['train'].num_examples // bs) * decay_epochs
min_lr = tf.keras.optimizers.schedules.ExponentialDecay(
    min_start_lr, decay_steps, min_decay, staircase=True
)
max_lr = tf.keras.optimizers.schedules.ExponentialDecay(
    max_start_lr, decay_steps, max_decay, staircase=True
)
quant_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=min_lr),
    loss_fn=tf.keras.losses.CategoricalCrossentropy(True),
    metric_clss=[tf.keras.metrics.CategoricalAccuracy],
    quant_forward=False,
    max_opt=tf.keras.optimizers.Adam(learning_rate=max_lr),
    lamda=lamda,
)
history = quant_model.fit(
        train_batches,
        epochs=n_epochs,
        validation_data=test_batches,
        # callbacks=[tensorboard, tensorboard_extra],
        callbacks=[tensorboard],
    )


# %%

