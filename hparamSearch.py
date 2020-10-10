# %%
import tensorflow as tf
import numpy as np
import larq as lq
from pathlib import Path
import datetime
from tensorboard.plugins.hparams import api as hp

from misc import get_datasets, get_cnn, TensorBoardExtra, MaybeSteSign
from quantizedModel import QuantModel


# %%
def train_w_hparams(hparams, log_dir):
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    tensorboard_extra = TensorBoardExtra(log_dir=log_dir)

    layers = get_cnn(input_shape, lambda: MaybeSteSign(to_quant=should_quantize), num_classes)
    quant_model = QuantModel(layers, should_quantize)

    decay_steps = (metadata.splits['train'].num_examples // bs) * hparams[DECAY_EPOCHS]
    min_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        hparams[MIN_START_LR], decay_steps, hparams[MIN_DECAY], staircase=True
    )
    max_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        hparams[MAX_START_LR], decay_steps, hparams[MAX_DECAY], staircase=True
    )
    quant_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=min_lr),
        loss_fn=tf.keras.losses.CategoricalCrossentropy(True),
        metric_clss=[tf.keras.metrics.CategoricalAccuracy],
        quant_forward=False,
        max_opt=tf.keras.optimizers.Adam(learning_rate=max_lr),
        lamda=hparams[LAMDA],
    )
    return quant_model.fit(
            train_batches,
            epochs=hparams[N_EPOCHS],
            validation_data=test_batches,
            callbacks=[tensorboard, tensorboard_extra],
        )

# %%
def run(log_dir, run_name, hparams):
    writer = tf.summary.create_file_writer(log_dir + '/hparam_tuning/' + run_name)
    writer.set_as_default()
    hp.hparams(hparams)  # record the values used in this trial
    trained_model = train_w_hparams(hparams, log_dir +'/' + run_name)
    writer.set_as_default()
    tf.summary.scalar(METRIC_ACCURACY, trained_model.history['val_q_CategoricalAccuracy'][-1], step=1)

# %%
should_quantize = tf.Variable(True, trainable=False, dtype=tf.bool, name='shuold_quant')
dataset = 'mnist'
bs = 256

train_batches, test_batches, metadata = get_datasets(dataset, bs)
input_shape = metadata.features['image'].shape
num_classes = metadata.features['label'].num_classes
# %%
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

MIN_START_LR = hp.HParam('min_start_lr', hp.RealInterval(4e-3,8e-3))
MAX_START_LR = hp.HParam('max_start_lr', hp.RealInterval(5e-4,4e-3))
MIN_DECAY = hp.HParam('min_decay', hp.RealInterval(0.85,0.9))
MAX_DECAY = hp.HParam('max_decay', hp.RealInterval(1.,1.8))
LAMDA = hp.HParam('lamda', hp.RealInterval(5e2,1e3))
N_EPOCHS = hp.HParam('n_epochs', hp.RealInterval(5e2,1e3))
DECAY_EPOCHS = hp.HParam('decay_epochs', hp.RealInterval(5e2,1e3))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer(log_dir + '/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[MIN_START_LR, MAX_START_LR, MIN_DECAY, MAX_DECAY, LAMDA],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )
# %%
n_runs = 20
n_epochs = 40
decay_epochs = 8

for session_num in range(n_runs):

    min_start_lr = MIN_START_LR.domain.sample_uniform()
    max_start_lr = MAX_START_LR.domain.sample_uniform()
    min_decay = MIN_DECAY.domain.sample_uniform()
    max_decay = MAX_DECAY.domain.sample_uniform()
    lamda = LAMDA.domain.sample_uniform()

    hparams = {
        MIN_START_LR: min_start_lr,
        MAX_START_LR: max_start_lr,
        MIN_DECAY: min_decay ,
        MAX_DECAY: max_decay ,
        LAMDA: lamda,
        N_EPOCHS: n_epochs,
        DECAY_EPOCHS: decay_epochs,

    }
    run_name = "run-%d" % session_num
    print('--- Starting trial: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    run(log_dir,  run_name, hparams)


# %%

