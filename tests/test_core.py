import numpy as np
import pytest

from core import Variable, Function, Square, Exp


class TestVariable:
    def test_init(self):
        x = Variable(np.array([1, 2, 3]))

        assert np.array_equal(x.data, np.array([1, 2, 3]))

    def test_assignment(self):
        x = Variable(np.array([1]))
        x.data = np.array([4, 5, 6])

        assert np.array_equal(x.data, np.array([4, 5, 6]))


class TestFunction:
    def test_func_instantiation(self):
        with pytest.raises(TypeError):
            Function()

    def test_square_call(self):
        x = Variable(np.array([3, 4, 5]))
        f = Square()
        y = f(x)

        assert isinstance(y, Variable)
        assert np.array_equal(y.data, np.array([9, 16, 25]))

    def test_func_composition(self):
        x = Variable(np.array(0.5))
        a = Square()(x)
        b = Exp()(a)
        y = Square()(b)

        assert np.allclose(y.data, np.array(1.64872127))
