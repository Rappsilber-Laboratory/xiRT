import numpy as np
from xiML import basic

def test_add_numbers():
    """
    Test addition of two numbers.
    :return:
    """
    x = np.array([10])
    y = np.array([30])
    result = basic.add_numbers(x, y)
    assert result == np.array([40])