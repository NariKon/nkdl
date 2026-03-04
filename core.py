from abc import ABC, abstractmethod
from typing import override
import numpy as np


class Variable:
    def __init__(self, data: np.generic | np.ndarray, creator: Function | None = None):
        if data is not None:
            if np.isscalar(data):
                data = np.array(data)
            elif not isinstance(data, np.ndarray):
                raise TypeError(
                    f"{type(data)} is not supported. Use np.ndarray instead."
                )

        self.data: np.ndarray = data
        self._creator: Function | None = creator
        self.grad: np.ndarray | None = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self._creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x._creator is not None:
                funcs.append(x._creator)


class Function(ABC):
    def __call__(self, input: Variable) -> Variable:
        self.input = input
        x = input.data
        y = self.forward(x)
        output = Variable(y, creator=self)
        self.output = output
        return output

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise TypeError()

    @abstractmethod
    def backward(self, gy: np.ndarray) -> np.ndarray:
        raise TypeError()


class Square(Function):
    @override
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.square(x)

    @override
    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        return 2 * x * gy


class Exp(Function):
    @override
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    @override
    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        return np.exp(x) * gy
