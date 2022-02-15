import tensorflow as tf
from tensorflow.keras import backend
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error


def keras_r2_score(y_true, y_pred):
    """Calculate the coefficient of determination (R^2) for Keras models"""
    return tf.py_function(r2_score, [tf.cast(y_true, tf.float64), tf.cast(y_pred, tf.float64)], Tout=tf.float64)


def keras_pearson(y_true, y_pred):
    """Calculate the Pearson correlation coefficient for Keras models"""
    # copied from: https://stackoverflow.com/a/46620771 (correlation coefficient loss, removing the modification they added to make it a loss)
    # implementation is similar to scipy's pearsonr and they say it returns the same result
    x = y_true
    y = y_pred
    mx = backend.mean(x)
    my = backend.mean(y)
    xm, ym = x - mx, y - my
    r_num = backend.sum(tf.multiply(xm, ym))
    r_den = backend.sqrt(tf.multiply(backend.sum(backend.square(xm)), backend.sum(backend.square(ym))))
    r = r_num / r_den
    r = backend.maximum(backend.minimum(r, 1.0), -1.0)
    return r


def keras_spearman(y_true, y_pred):
    """Calculate the Spearman correlation coefficient for Keras models"""
    return tf.py_function(spearman, [tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)], Tout=tf.float32)


def spearman(y, y_pred):
    """Calculate the Spearman correlation coefficient"""
    return spearmanr(y, y_pred, nan_policy='omit')[0]  # omit = ignores NaNs


def pearson(y, y_pred):
    """Calculate the Pearson correlation coefficient"""
    return pearsonr(y, y_pred)[0]
