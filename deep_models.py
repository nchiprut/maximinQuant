
# %%
import tensorflow as tf
import numpy as np
import larq as lq

from misc import get_convnet_layers, MaybeSteSign, get_optimizer
from quantizedModel import QuantModel

# %%
def model_metargs(name):
    return { 
        'maximin': {
            'max_op': True,
            'quant_forward': False,
            'kernel_constraint': None,
        },
        'maximin_clip': {
            'max_op': True,
            'quant_forward': False,
            'kernel_constraint': 'weight_clip',
        },
        'maximin_ste': {
            'max_op': True,
            'quant_forward': True,
            'kernel_constraint': 'weight_clip',
        },
        'ste': {
            'max_op': False,
            'quant_forward': True,
            'kernel_constraint': 'weight_clip',
        },
        'real': {
            'max_op': False,
            'quant_forward': False,
            'kernel_constraint': None,
        },
    }[name]

def get_quant_model(name, quant_var, args, fc=(), conv=()):
    args.update(**model_metargs(name))
    quant_getter = lambda: MaybeSteSign(to_quant=quant_var)
    layers = get_convnet_layers(args.input_shape, args.n_classes, quant_getter, args.kernel_constraint, fc, conv)
    quant_model = QuantModel(layers, quant_var)

    min_opt = get_optimizer(args.min_lr, args.min_decay, args.n_iter // args.n_decay)
    if args.max_op:
        max_opt = get_optimizer(args.max_lr, args.max_decay, args.n_iter // args.n_decay)
    else:
        max_opt = None

    quant_model.compile(
        optimizer=min_opt,
        loss_fn=tf.keras.losses.CategoricalCrossentropy(True),
        metric_clss=[tf.keras.metrics.CategoricalAccuracy],
        quant_forward=args.quant_forward,
        max_opt=max_opt,
        lamda=args.lamda,
    )
    return quant_model
