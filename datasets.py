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


def boston_data(split=.3,seed=None, p_out=0, out_val=100.):
    X, y = load_boston(return_X_y=True)
    # X, y = load_diabetes(return_X_y=True)
    y = y[:,np.newaxis]
    n, d = X.shape
    n_test = int(n*split)
    n_train = n - n_test
    np.random.seed(seed)
    test_ind = np.random.choice(n,n_test)
    test_msk = np.zeros(n, dtype=bool)
    test_msk[test_ind] = True

    X_train = X[~test_msk]
    y_train = y[~test_msk]
    X_test = X[test_msk]
    y_test = y[test_msk]
    if p_out:
        n_out = int(p_out*n_train)
        np.random.seed(seed+1)
        out_ind = np.random.choice(n_train,n_out)
        X_train = np.block([[X_train[out_ind]], [X_train]])
        y_train = np.block([ [np.ones((n_out,1))*out_val], [y_train] ])

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    y_mean = y_train.mean()
    y_std = y_train.std()

    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    return X_train, y_train, X_test, y_test, np.zeros(d)
