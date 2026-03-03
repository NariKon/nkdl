from abc import ABC, abstractmethod
import numpy as np


class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data


class Function(ABC):
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        return Variable(y)

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise TypeError()


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.square(x)


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)
