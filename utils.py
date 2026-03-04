from typing import Callable
from core import Variable


def numerical_diff(f: Callable[[Variable], Variable], x: Variable, eps=1e-4):
    y0 = f(Variable(x.data - eps))
    y1 = f(Variable(x.data + eps))
    return (y1.data - y0.data) / (2 * eps)
