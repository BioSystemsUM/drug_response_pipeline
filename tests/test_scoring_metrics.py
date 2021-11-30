import numpy as np
import tensorflow as tf
from src.scoring.scoring_metrics import *

def test_keras_pearson(a, b):
    assert np.isclose(keras_pearson(tf.convert_to_tensor(a, dtype=tf.float32), tf.convert_to_tensor(b, dtype=tf.float32)).numpy(), pearson(a, b), atol=1e-4, equal_nan=True)

if __name__ == '__main__':
    arr1 = np.array([0, 0, 0, 1, 1, 1, 1])
    arr2 = np.arange(7)
    test_keras_pearson(arr1, arr2)

    arr3 = np.array([1, 1, 1, 1, 1, 1, 1])
    arr4 = np.arange(7)
    test_keras_pearson(arr3, arr4)