# %%
# %%
import time
import cvxpy as cp
import numpy as np
import tensorflow as tf
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from scipy.optimize import lsq_linear
from quantizedModel import QuantModel


def run_cvxpy(problem, solver):
    try:
        problem.solve(solver=solver)
    except cp.error.SolverError as err:
        try:
            problem.solve(solver=solver, verbose=True)
        except cp.error.SolverError as err:
            print("cvxpy error: {0}".format(err))

def unconstrained(y, X, obj_loss, seed):
    start = time.time()
    n, d = np.shape(X)
    w = cp.Variable((d, 1))
    obj = cp.Minimize(cp.sum(obj_loss(y, X, w, cp)))
    prob = cp.Problem(obj)
    run_cvxpy(prob, cp.ECOS)

    print('unconstrained', 'loss: ', obj_loss.__name__, ', number of samples: ', X.shape[0], 'time', time.time() - start)
    return w.value

def sdr(y, X, obj_loss, seed):
    start = time.time()
    n, d = np.shape(X)
    w = cp.Variable((d+1, d+1))
    constraints = [cp.diag(w) == 1, w >> 0]
    obj = cp.Minimize(cp.sum(obj_loss(y, X, w, cp)))
    prob = cp.Problem(obj, constraints)
    run_cvxpy(prob, cp.SCS)

    U, *_ = np.linalg.svd(w.value)
    w_val = U[:-1, :1] / U[-1, 0]
    print('sdr', 'loss: ', obj_loss.__name__, ', number of samples: ', X.shape[0], 'time', time.time() - start)
    return w_val

def lpr(y, X, obj_loss, seed):
    start = time.time()
    n, d = np.shape(X)
    w = cp.Variable((d, 1))
    constraints = [w <= 1, w >= -1]
    obj = cp.Minimize(cp.sum(obj_loss(y, X, w, cp)))
    prob = cp.Problem(obj, constraints)
    run_cvxpy(prob, cp.ECOS)

    print('lpr', 'loss: ', obj_loss.__name__, ', number of samples: ', X.shape[0], 'time', time.time() - start)
    return w.value

def scipy_lpr(y, X, obj_loss, seed):
    start = time.time()
    d = X.shape[1]
    regr = lsq_linear(X, y.squeeze(), bounds=(-np.ones(d), np.ones(d)))
    print('scipy lpr', 'loss: ', obj_loss.__name__, ', number of samples: ', X.shape[0], 'time', time.time() - start)
    return regr.x[:, np.newaxis]

def sklearn_lr(y, X, obj_loss, seed):
    start = time.time()
    regr = LinearRegression(fit_intercept=False)
    regr.fit(X,y)
    print('sklearn lr', 'loss: ', obj_loss.__name__, ', number of samples: ', X.shape[0], 'time', time.time() - start)
    return regr.coef_.T

def sklearn_huber_regression(y, X, obj_loss, seed):
    start = time.time()
    regr = HuberRegressor(fit_intercept=False)
    regr.fit(X,y)
    print('sklearn rlr', 'loss: ', obj_loss.__name__, ', number of samples: ', X.shape[0], 'time', time.time() - start)
    return regr.coef_.T

def ste(y, X, obj_loss, seed, n_iter, lr=1e-2):
    start = time.time()
    n, d = np.shape(X)
    w = tf.Variable(1e-2*np.random.normal(size=(d,1)), name='w', dtype=tf.float32)

    w_opt = tf.keras.optimizers.SGD(learning_rate=lr)

    for step in range(n_iter):
        with tf.GradientTape() as w_tape:
            w_quant = w + tf.stop_gradient(tf.sign(w) - w)
            loss = tf.reduce_sum(obj_loss(y, X, w_quant, tf))

        w_gradients = w_tape.gradient(loss, [w])
        w_opt.apply_gradients(zip(w_gradients, [w]))

    #     if (step % 100 == 0):
    #         print (loss.numpy())
    # print ('----')
    print('ste', 'loss: ', obj_loss.__name__, ', number of samples: ', X.shape[0], 'time', time.time() - start)
    return w.numpy()

def maxmin(y, X, obj_loss, seed, n_iter, lr=1e-2, max_decay=1.5, min_decay=0.8, d_epochs=5):
    start = time.time()

    n, d = np.shape(X)
    w = tf.Variable(1e-2*np.random.normal(size=(d,1)), name='w', dtype=tf.float32)
    z = tf.Variable(1e-2*np.random.normal(size=(d,1)), name='z', dtype=tf.float32)

    max_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        lr, n_iter // d_epochs, max_decay, staircase=True
    )
    min_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        lr, n_iter // d_epochs, min_decay, staircase=True
    )

    w_opt = tf.keras.optimizers.Adam(learning_rate=min_lr)
    z_opt = tf.keras.optimizers.Adam(learning_rate=max_lr)

    for step in range(n_iter):
        with tf.GradientTape() as w_tape:
            primary_loss = tf.reduce_sum(obj_loss(y, X, w, tf))
            w_loss = primary_loss + tf.reduce_sum(z * (1 - w ** 2)) 

        # primary_loss_rnd = tf.reduce_sum(obj_loss(y, X, tf.sign(w), tf))
        w_gradients = w_tape.gradient(w_loss, [w])
        w_opt.apply_gradients(zip(w_gradients, [w]))

        with tf.GradientTape() as z_tape:
            z_loss = -tf.reduce_sum(z * (1 - w ** 2))

        z_gradients = z_tape.gradient(z_loss, [z])
        z_opt.apply_gradients(zip(z_gradients, [z]))

    #     if (step % (n_iter // 10) == 0):
    #         print (f'step: {step}, loss: {primary_loss.numpy()}, rounded: {primary_loss_rnd.numpy()}')
            
    # print ('----')
    print('maximin', 'loss: ', obj_loss.__name__, ', number of samples: ', X.shape[0], 'time', time.time() - start)
    return w.numpy()


def brute_force(y, X, obj_loss):
    cands = [np.reshape(w, (X.shape[1], -1)) for w in itertools.product([-1,1], repeat=X.shape[1])]
    i = np.argmin([np.sum(obj_loss(y, X, w, np)) for w in cands])
    return cands[i]

