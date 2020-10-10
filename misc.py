# %%
import os
import yaml
import tensorflow as tf
import numpy as np
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
        if self.model.max_opt is not None:
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

def print_evals_stat(evals):
    for metric in evals[0].keys():
        print(f'metric: {metric}')
        for k in evals[0][metric].keys():
            print(f'method: {k}', end='\t')
            curr = [evals[i][metric][k][0] for i in range(len(evals))]
            print(f'mean: {np.mean(curr)} \t std: {np.std(curr)}')

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

def get_convnet_layers(input_shape, num_classes, quant_getter, kernel_constraint=None, fc=(), conv=()):
    ret = [tf.keras.layers.InputLayer(input_shape=input_shape)]
    max_pool_all = 2**len(conv) < input_shape[0]
    for i, conv_i in enumerate(conv):
        ret.append(
            lq.layers.QuantConv2D(conv_i, (3, 3), use_bias=False,
                                  kernel_quantizer=quant_getter(),
                                  kernel_constraint=kernel_constraint,
                                  padding='SAME')
        )
        if max_pool_all or (i % 2 == 1):
            ret.append(tf.keras.layers.MaxPooling2D((2, 2)))
        ret += [tf.keras.layers.BatchNormalization(scale=True),
                tf.keras.layers.ReLU()]

    ret.append(tf.keras.layers.Flatten())
    for fc_i in fc:
        ret.append(
            lq.layers.QuantDense(fc_i,
                                 kernel_quantizer=quant_getter(),
                                 kernel_constraint=kernel_constraint,
                                 use_bias=False)
        )
        ret += [tf.keras.layers.BatchNormalization(scale=True),
                tf.keras.layers.ReLU()]

    ret.append(
            lq.layers.QuantDense(num_classes,
                                 kernel_quantizer=quant_getter(),
                                 kernel_constraint=kernel_constraint,
                                 use_bias=False)
    )
    return ret

def get_optimizer(start_lr, decay, decay_steps):

    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        start_lr, decay_steps, decay, staircase=True
    )
    return tf.keras.optimizers.Adam(learning_rate=lr)


class Bunch(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def update(self, **kwds):
        self.__dict__.update(kwds)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__