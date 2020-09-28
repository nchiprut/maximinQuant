# %%
import numpy as np
from sklearn.datasets import load_boston
import tensorflow as tf
import tensorflow_datasets as tfds

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

def img_data(ds, bs, shuffle_size=1000, augment=False):

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

    train_pre= lambda img, label: _preprocess(img, label, mean, std, num_classes, augment=augment)
    test_pre= lambda img, label: _preprocess(img, label, mean, std, num_classes, augment=False)
    train_batches = raw_train.map(train_pre).shuffle(shuffle_size,seed=1).batch(bs).prefetch(tf.data.experimental.AUTOTUNE)
    test_batches = raw_test.map(test_pre).batch(bs).prefetch(tf.data.experimental.AUTOTUNE)

    return train_batches, test_batches, metadata

# def _preprocess(img, label, mean, std, num_classes, augment=False):
#     img = (tf.cast(img, tf.float32) - mean) / std
#     label = tf.one_hot(tf.squeeze(label), num_classes)
#     return img, label

def _preprocess(img, label, mean, std, num_classes, augment=False):
    img = (tf.cast(img, tf.float32) - mean) / std
    label = tf.one_hot(tf.squeeze(label), num_classes)
    if augment:
        img = _augmentation(img)
    img = tf.image.per_image_standardization(img)
    return img, label

def _augmentation(img):
    img_size = img.get_shape().as_list()
    h, w, c = img.shape.as_list()


    img = tf.image.random_crop(img, [h - 3, w - 3, c])
    img = tf.image.resize_with_crop_or_pad(img, h, w)
    img = tf.image.random_flip_left_right(img)

    img = tf.image.random_brightness(img, max_delta=63)
    img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
    return img