import numpy as np

from core import Variable, Square, Exp
from utils import numerical_diff


def test_numerical_diff_of_square():
    f = Square()  # x^2
    x = Variable(np.array(2.0))

    dy = numerical_diff(f, x)

    assert np.isclose(dy, 2 * x.data)  # 2x


def test_numerical_diff_of_composit_func():
    f = lambda x: Square()(Exp()(Square()(x)))  # (e^{x^2})^2 = e^{2x^2}
    x = Variable(np.array(0.5))

    dy = numerical_diff(f, x)

    assert np.isclose(dy, 4 * x.data * np.e ** (2 * (x.data**2)))  # 4xe^{2x^2}
