from abc import ABC, abstractmethod
from collections.abc import Sequence
import heapq
from typing import override
import weakref
import numpy as np

from config import Config


class Variable:
    def __init__(self, data: np.generic | np.ndarray, name: str | None = None):
        if data is not None:
            if np.isscalar(data):
                data = np.array(data)
            elif not isinstance(data, np.ndarray):
                raise TypeError(
                    f"{type(data)} is not supported. Use np.ndarray instead."
                )

        self.data = data
        self.name = name
        self._creator: Function | None = None
        self.generation = 0
        self.grad: np.ndarray | None = None

    def backward(self, retain_grad=False):
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
            gys = [output().grad for output in f.outputs]
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

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self):
        self.grad = None

    def set_creator(self, creator: Function):
        self._creator = creator
        self.generation = creator.generation + 1

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        return f'variable({str(self.data).replace('\n', '\n' + ' ' * 9)})'


class Function(ABC):
    # TODO: Sequence[Variable] -> Sequence[Variable] is preferable
    def __call__(self, *inputs: Sequence[Variable]) -> Variable | Sequence[Variable]:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(y) for y in ys]
        if Config.enable_backprop:
            self.inputs = inputs
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.outputs = [weakref.ref(output) for output in outputs]

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
