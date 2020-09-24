# %%
# %%
import cvxpy as cp
import numpy as np
import tensorflow as tf
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
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
    print('unconstrained', 'loss: ', obj_loss.__name__, ', number of samples: ', X.shape[0])
    n, d = np.shape(X)
    w = cp.Variable((d, 1))
    obj = cp.Minimize(cp.sum(obj_loss(y, X, w, cp)))
    prob = cp.Problem(obj)
    run_cvxpy(prob, cp.ECOS)

    return w.value

def sdr(y, X, obj_loss, seed):
    print('sdr', 'loss: ', obj_loss.__name__, ', number of samples: ', X.shape[0])
    n, d = np.shape(X)
    w = cp.Variable((d+1, d+1))
    constraints = [cp.diag(w) == 1, w >> 0]
    obj = cp.Minimize(cp.sum(obj_loss(y, X, w, cp)))
    prob = cp.Problem(obj, constraints)
    run_cvxpy(prob, cp.SCS)

    U, *_ = np.linalg.svd(w.value)
    w_val = U[:-1, :1] / U[-1, 0]
    return w_val

def lpr(y, X, obj_loss, seed):
    print('lpr', 'loss: ', obj_loss.__name__, ', number of samples: ', X.shape[0])
    n, d = np.shape(X)
    w = cp.Variable((d, 1))
    constraints = [w <= 1, w >= -1]
    obj = cp.Minimize(cp.sum(obj_loss(y, X, w, cp)))
    prob = cp.Problem(obj, constraints)
    run_cvxpy(prob, cp.ECOS)

    return w.value

def sklearn_lr(y, X, obj_loss, seed):
  regr = LinearRegression(fit_intercept=False)
  regr.fit(X,y)
  return regr.coef_

def sklearn_huber_regression(y, X, obj_loss, seed):
  regr = HuberRegressor(fit_intercept=False)
  regr.fit(X,y)
  return regr.coef_

def maxmin(y, X, obj_loss, seed, n_iter):

    max_lr=1e-2
    min_lr=1e-2
    n, d = np.shape(X)
    w = tf.Variable(1e-2*np.random.normal(size=(d,1)), name='w', dtype=tf.float32)
    z = tf.Variable(1e-2*np.random.normal(size=(d,1)), name='z', dtype=tf.float32)

    max_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        max_lr, n_iter // 10, 2., staircase=True
    )

    w_opt = tf.keras.optimizers.SGD(learning_rate=min_lr)
    z_opt = tf.keras.optimizers.SGD(learning_rate=max_lr)

    for step in range(n_iter):
        with tf.GradientTape() as w_tape:
            primary_loss = tf.reduce_sum(obj_loss(y, X, w, tf))
            w_loss = primary_loss + tf.reduce_sum(z * (1 - w ** 2)) 

        w_gradients = w_tape.gradient(w_loss, [w])
        w_opt.apply_gradients(zip(w_gradients, [w]))

        with tf.GradientTape() as z_tape:
            z_loss = -tf.reduce_sum(z * (1 - w ** 2))

        z_gradients = z_tape.gradient(z_loss, [z])
        z_opt.apply_gradients(zip(z_gradients, [z]))

        if (step % 100 == 0):
            print (primary_loss.numpy())
    print ('----')
    return w.numpy()


def brute_force(y, X, obj_loss):
    cands = [np.reshape(w, (X.shape[1], -1)) for w in itertools.product([-1,1], repeat=X.shape[1])]
    i = np.argmin([np.sum(obj_loss(y, X, w, np)) for w in cands])
    return cands[i]

