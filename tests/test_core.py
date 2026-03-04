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


class TestBackPropagation:
    def test_manual_back_propagation(self):
        A = Square()
        B = Exp()
        C = Square()

        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)  # (e^{x^2})^2 = e^{2x^2}

        y.grad = np.array(1.0)
        b.grad = C.backward(y.grad)
        a.grad = B.backward(b.grad)
        x.grad = A.backward(a.grad)

        assert np.isclose(x.grad, 4 * x.data * np.e ** (2 * (x.data**2)))  # 4xe^{2x^2}

    def test_traverse_linked_nodes_in_reverse(self):
        A = Square()
        B = Exp()
        C = Square()

        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)

        assert y._creator == C
        assert y._creator.input == b
        assert y._creator.input._creator == B
        assert y._creator.input._creator.input == a
        assert y._creator.input._creator.input._creator == A
        assert y._creator.input._creator.input._creator.input == x

    def test_automated_back_propagation(self):
        A = Square()
        B = Exp()
        C = Square()

        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)

        # y.grad = np.array(1.0)
        # `backward` method automatically set ones-like array for `y`
        y.backward()
        assert np.isclose(x.grad, 4 * x.data * np.e ** (2 * (x.data**2)))  # 4xe^{2x^2}
