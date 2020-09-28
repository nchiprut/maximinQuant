# %%
import tensorflow as tf
import numpy as np
import larq as lq
from pathlib import Path
import datetime

from datasets import img_data
from misc import get_convnet_layers, Bunch, TensorBoardExtra, MaybeSteSign
from deep_models import model_metargs, get_quant_model
from quantizedModel import QuantModel


# %%
model_args = Bunch(
    n_epochs=40,
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
    name='mnist',
    bs=256,
)
fc = ()
conv = ()
base_log_dir = "logs/mnist/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

quant_var = tf.Variable(True, trainable=False, dtype=tf.bool, name='shuold_quant')

train_batches, test_batches, metadata = img_data(ds_args.name, ds_args.bs)
model_args.input_shape = metadata.features['image'].shape
model_args.n_classes = metadata.features['label'].num_classes
model_args.n_iter = (metadata.splits['train'].num_examples // ds_args.bs) * model_args.n_epochs

# %%
log_dir = base_log_dir + '/maximin'
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
tensorboard_extra = TensorBoardExtra(log_dir=log_dir)
maximin_model = get_quant_model('maximin', quant_var, model_args, fc=fc, conv=conv)
history = maximin_model.fit(
        train_batches,
        epochs=model_args.n_epochs,
        validation_data=test_batches,
        callbacks=[tensorboard, tensorboard_extra],
    )

# %%
log_dir = base_log_dir + '/real'
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
tensorboard_extra = TensorBoardExtra(log_dir=log_dir)
real_model = get_quant_model('real', quant_var, model_args, fc=fc, conv=conv)
history = real_model.fit(
        train_batches,
        epochs=model_args.n_epochs,
        validation_data=test_batches,
        callbacks=[tensorboard, tensorboard_extra],
    )
# %%
log_dir = base_log_dir + '/ste'
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
tensorboard_extra = TensorBoardExtra(log_dir=log_dir)
ste_model = get_quant_model('ste', quant_var, model_args, fc=fc, conv=conv)
history = ste_model.fit(
        train_batches,
        epochs=model_args.n_epochs,
        validation_data=test_batches,
        callbacks=[tensorboard, tensorboard_extra],
    )

# %%
