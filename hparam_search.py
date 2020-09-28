
# %%
import tensorflow as tf
import numpy as np
import larq as lq
from pathlib import Path
import datetime
from tensorboard.plugins.hparams import api as hp

from datasets import img_data
from misc import get_convnet_layers, Bunch, TensorBoardExtra, MaybeSteSign
from deep_models import model_metargs, get_quant_model
from quantizedModel import QuantModel


# %%
model_args = Bunch(
    n_epochs=100,
    n_decay=5,
    min_decay=0.9,
    max_decay=1.5,
    min_lr=5e-3,
    max_lr=2e-3,
    lamda=1e3,
    #n_iter
    #n_classes
    #input_shape
)
ds_args = Bunch(
    name='cifar10',
    augment=True,
    bs=128,
)
fc = (1024, 1024)
conv = (128, 128, 256, 256, 512, 512,)
base_log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

quant_var = tf.Variable(True, trainable=False, dtype=tf.bool, name='shuold_quant')

train_batches, test_batches, metadata = img_data(ds_args.name, ds_args.bs, augment=ds_args.augment)
model_args.input_shape = metadata.features['image'].shape
model_args.n_classes = metadata.features['label'].num_classes
model_args.n_iter = (metadata.splits['train'].num_examples // ds_args.bs) * model_args.n_epochs

MIN_START_LR = hp.HParam('min_lr', hp.RealInterval(4e-3,8e-3))
MAX_START_LR = hp.HParam('max_lr', hp.RealInterval(5e-4,4e-3))
MIN_DECAY = hp.HParam('min_decay', hp.RealInterval(0.85,0.9))
MAX_DECAY = hp.HParam('max_decay', hp.RealInterval(1.,1.8))
LAMDA = hp.HParam('lamda', hp.RealInterval(5e2,1e3))
N_DECAY = hp.HParam('n_decay', hp.IntInterval(5,20))

METRIC_ACCURACY = 'accuracy'

log_dir = base_log_dir + '/maximin'
with tf.summary.create_file_writer(log_dir + '/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[MIN_START_LR, MAX_START_LR, MIN_DECAY, MAX_DECAY, N_DECAY, LAMDA],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )
# %%
n_runs = 20

for session_num in range(n_runs):

    hparams = {
        MIN_START_LR: MIN_START_LR.domain.sample_uniform(),
        MAX_START_LR: MAX_START_LR.domain.sample_uniform(),
        MIN_DECAY: MIN_DECAY.domain.sample_uniform(),
        MAX_DECAY: MAX_DECAY.domain.sample_uniform(),
        LAMDA: LAMDA.domain.sample_uniform(),
        N_DECAY: N_DECAY.domain.sample_uniform(),

    }
    model_args.update(**{k.name: v for k,v in hparams.items()})
    run_name = '_'.join([f'{k.name}-{v}' for k, v in hparams.items()])
    print(f'--- Starting trial: {run_name} ---')

    writer = tf.summary.create_file_writer(log_dir + '/hparam_tuning/' + run_name)
    writer.set_as_default()
    hp.hparams(hparams)  # record the values used in this trial

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir + '/' + run_name, histogram_freq=10)
    tensorboard_extra = TensorBoardExtra(log_dir=log_dir + '/' + run_name)
    maximin_model = get_quant_model('maximin', quant_var, model_args, fc=fc, conv=conv)
    history = maximin_model.fit(
            train_batches,
            epochs=model_args.n_epochs,
            validation_data=test_batches,
            callbacks=[tensorboard, tensorboard_extra],
        )

    writer.set_as_default()
    tf.summary.scalar(METRIC_ACCURACY, np.mean(history.history['val_q_CategoricalAccuracy'][-5:]), step=1)



# %%

