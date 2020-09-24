# %%
# %%
import numpy as np
from sklearn.datasets import load_boston

def regression_data(d, var, noise_fn, n_train=5000, n_test=None, w_star=None, seed=0):

    if n_test is None:
        n_test = n_train
    np.random.seed(seed)
    X1 = np.sqrt(1./n_train)*np.random.randn(n_train, d)
    np.random.seed(seed+1)
    X2 = np.sqrt(1./n_train)*np.random.randn(n_test, d)
    X = np.block([[X1],[X2]])
    np.random.seed(seed+2)
    if w_star is None:
        w_star = np.sign(np.random.randn(d, 1))
    np.random.seed(seed+3)
    eps = np.concatenate((noise_fn(size=(n_train, 1)), noise_fn(size=(n_test, 1))))
    y = np.cast[np.float32](X @ w_star + (np.sqrt(var)*eps))
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:], w_star


def boston_data(split=406,seed=None, n_out=0):
    X, y = load_boston(return_X_y=True)
    if n_out:
        out_ind = np.random.choice(split,n_out)
        out_val = -10
        X = np.block([[X[out_ind]], [X]])
        y = np.concatenate((np.ones(n_out)*out_val, y))
        split += n_out
    y = y[:,np.newaxis]
    return X[:split], y[:split], X[split:], y[split:], np.zeros(X.shape[1])
