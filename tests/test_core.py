import numpy as np

from core import Variable


class TestVariable:
    def test_init(self):
        x = Variable(np.array([1, 2, 3]))

        assert np.array_equal(x.data, np.array([1, 2, 3]))

    def test_assignment(self):
        x = Variable(np.array([1]))
        x.data = np.array([4, 5, 6])

        assert np.array_equal(x.data, np.array([4, 5, 6]))
