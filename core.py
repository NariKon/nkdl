from abc import ABC, abstractmethod
from collections.abc import Sequence
import heapq
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
        self.generation = creator.generation + 1 if creator is not None else 0
        self.grad: np.ndarray | None = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = list()
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                heapq.heappush_max(funcs, f)
                seen_set.add(f)

        add_func(self._creator)
        while funcs:
            f = heapq.heappop_max(funcs)
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    # create new instance since in-place operation would modify the previous Variable's grad
                    x.grad = x.grad + gx

                if x._creator is not None:
                    add_func(x._creator)

    def cleargrad(self):
        self.grad = None


class Function(ABC):
    # TODO: Sequence[Variable] -> Sequence[Variable] is preferable
    def __call__(self, *inputs: Sequence[Variable]) -> Variable | Sequence[Variable]:
        self.inputs = inputs
        self.generation = max([x.generation for x in inputs])

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(y, creator=self) for y in ys]
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def __lt__(self, other: Function):
        return self.generation < other.generation

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
        x = self.inputs[0].data
        return 2 * x * gy


class Exp(Function):
    @override
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    @override
    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        return np.exp(x) * gy


class Add(Function):
    @override
    def forward(self, x1: np.ndarray, x2: np.ndarray) -> tuple[np.ndarray]:
        return (x1 + x2,)

    @override
    def backward(self, gy: np.ndarray) -> tuple[np.ndarray]:
        return (gy, gy)
