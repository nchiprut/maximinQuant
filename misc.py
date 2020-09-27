# %%
import os
import yaml
import tensorflow as tf
import tensorflow_datasets as tfds
import larq as lq
import larq_zoo as lqz
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

# %%
class TensorBoardExtra(Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        tf.summary.histogram('z', self.model.z, epoch)
        tf.summary.scalar('max_lr', self.model.max_opt._decayed_lr(tf.float32), epoch)
        tf.summary.scalar('min_lr', self.model.optimizer._decayed_lr(tf.float32), epoch)

    def on_train_begin(self, logs=None):
        file_writer = tf.summary.create_file_writer(self.log_dir + "/metrics")
        file_writer.set_as_default()
        
class MaybeSteSign(tf.keras.layers.Layer):
    def __init__(self, to_quant, *args, trainable=False, **kwargs):
        super().__init__(trainable=trainable, *args, **kwargs)
        class Dummy(object):
            def __init__(self, val):
                self.val = val

        self.to_quant = Dummy(to_quant)
    def call(self, inputs):
        return tf.cond(self.to_quant.val, lambda: lq.quantizers.SteSign()(inputs), lambda: inputs)

def get_datasets(ds, bs, shuffle_size=1000):

    (raw_train, raw_test), metadata = tfds.load(
        ds,
        split=[tfds.Split.TRAIN, tfds.Split.TEST],
        with_info=True,
        as_supervised=True,
    )

    num_classes = metadata.features['label'].num_classes
        
    stat_batch = tf.cast(list(iter(
        raw_train.shuffle(shuffle_size).batch(shuffle_size).take(1)
        ))[0][0], tf.float32)
    mean = tf.math.reduce_mean(stat_batch, axis=0)
    std = tf.math.reduce_std(stat_batch, axis=0)
    std += tf.cast(std==0, tf.float32)

    def preprocess(img, label):
        img = (tf.cast(img, tf.float32) - mean) / std
        label = tf.one_hot(tf.squeeze(label), num_classes)
        return img, label

    train_batches = raw_train.map(preprocess).shuffle(shuffle_size,seed=1).batch(bs).prefetch(tf.data.experimental.AUTOTUNE)
    test_batches = raw_test.map(preprocess).batch(bs).prefetch(tf.data.experimental.AUTOTUNE)

    return train_batches, test_batches, metadata

def get_fc(input_shape, quant_getter, num_classes, fc=()):
    return \
        [
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Flatten(),
        ] +  \
        [
            lq.layers.QuantDense(fc_i,
                                 kernel_quantizer=quant_getter(),
                                 #   kernel_quantizer='ste_sign',
                                 #  kernel_constraint="weight_clip",
                                 use_bias=False)
            for fc_i in fc

        ] +  \
        [
            lq.layers.QuantDense(num_classes,
                                 kernel_quantizer=quant_getter(),
                                 #  kernel_quantizer='ste_sign',
                                 #  kernel_constraint="weight_clip",
                                 use_bias=False)
        ]
def get_cnn(input_shape, quant_getter, num_classes):
    return \
    [
        tf.keras.layers.InputLayer(input_shape=input_shape),
        lq.layers.QuantConv2D(32, (3, 3), use_bias=False,
                            kernel_quantizer=quant_getter(),
                            padding='SAME',
                            ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(scale=True),
        tf.keras.layers.ReLU(),

        lq.layers.QuantConv2D(64, (3, 3), use_bias=False,
                            kernel_quantizer=quant_getter(),
                            padding='SAME',
                            ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(scale=True),
        tf.keras.layers.ReLU(),

        lq.layers.QuantConv2D(128, (3, 3), use_bias=False,
                            kernel_quantizer=quant_getter(),
                            padding='SAME',
                            ),
        tf.keras.layers.BatchNormalization(scale=True),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),

        lq.layers.QuantDense(128, use_bias=False,
                            kernel_quantizer=quant_getter(),
                            ),
        tf.keras.layers.BatchNormalization(scale=True),
        tf.keras.layers.ReLU(),
        lq.layers.QuantDense(128, use_bias=False,
                            kernel_quantizer=quant_getter(),
                            ),
        tf.keras.layers.BatchNormalization(scale=True),
        tf.keras.layers.ReLU(),
        lq.layers.QuantDense(128, use_bias=False,
                            kernel_quantizer=quant_getter(),
                            ),
        tf.keras.layers.BatchNormalization(scale=True),
        tf.keras.layers.ReLU(),
        lq.layers.QuantDense(num_classes, use_bias=False,
                            kernel_quantizer=quant_getter(),
                            )
    ]

def avg_evals(evals):
    # evals are list of dicts

    n_evals = sum([eval is not None for eval in evals])
    dest_eval = evals[0].copy()

    for metric, graphs in dest_eval.items():
        for method, vals in graphs.items():
            # plt.plot(steps, vals, label=method)
            for i in range(len(vals)):
                for eval in evals[1:]:
                    dest_eval[metric][method][i] += eval[metric][method][i]
                dest_eval[metric][method][i] /= n_evals
    return dest_eval

def dump_evals(path, name, evals):
    with open(os.path.join(path, name), 'w') as outfile:
        yaml.dump(evals, outfile, default_flow_style=False)

def plot_dicts(steps, dicts, dir, save=True, logscale=False):

    for metric, graphs in dicts.items():
        for method, vals in graphs.items():
            plt.plot(steps, vals, label=method)
        plt.title(metric)
        plt.legend()
        if logscale:
            plt.yscale('log')
        if save:
            plt.savefig(
                os.path.join(dir, metric + '.png')
            )
            plt.clf()
        else:
            plt.show()
