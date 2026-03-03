from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad: Optional[np.ndarray] = None


class Function(ABC):
    def __call__(self, input: Variable) -> Variable:
        self.input = input
        x = input.data
        y = self.forward(x)
        return Variable(y)

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise TypeError()

    @abstractmethod
    def backward(self, gy: np.ndarray) -> np.ndarray:
        raise TypeError()


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.square(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        return 2 * x * gy


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        return np.exp(x) * gy
