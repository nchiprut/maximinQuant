import cvxpy as cp
import numpy as np
import tensorflow as tf

def hinge(y, X, w, mod):
    return mod.maximum(0., 1 - (y * X) @ w)

def l1(y, X, w, mod):
    return mod.abs(X @ w - y)

def huber(y, X, w, mod, t=None):
    if t is None:
        t = 1e-1
    return mod.sqrt((X @ w - y)**2 + t)

def l2(y, X, w, mod):
    return (X @ w - y)**2

def matrix_linear(y, X, W, mod):
    Q = np.block([[X.T @ X,  -X.T @ y],
                 [-y.T @ X,  y.T @ y]])
    return mod.multiply(Q,W)

def logistic(y, X, w, mod):
    pred = -(y * X) @ w
    if mod is tf:
        return tf.reduce_logsumexp([tf.zeros(pred.shape,tf.float32), pred], axis=0)
    if mod is np:
        return np.logaddexp(0., pred)
    if mod is cp:
        return cp.logistic(pred)

def feas_dist(y, X, w, w_star=None):
    return np.mean(np.abs(w- np.sign((w)))).item()

def w_star_dist(y, X, w, w_star):
    return 0.5 * np.mean(np.abs(w - w_star)).item()
    # return np.mean((w_star==w))

def np_mean_loss(y, X, w, w_star=None, loss=l2):
    return np.mean(loss(y, X, w, np)).item()

def np_sum_loss(y, X, w, w_star=None, loss=l2):
    return np.sum(loss(y, X, w, np)).item()
      
def mean_loss(loss, mod, *args, **kwargs):
    if mod is tf:
        return tf.reduce_mean(loss(mod=mod, *args, **kwargs))
    else:
        return mod.mean(loss(mod=mod, *args, **kwargs))