import numpy as np
from memory_profiler import profile
from core import Variable
from functions import square
from config import Config


@profile
def nested_square_loop(iter_max=10):
    for _ in range(iter_max):
        x = Variable(np.array(10_000))
        y = square(square(square(x)))


@profile
def nested_square_3d_with_backprops():
    x = Variable(np.random.rand(100, 100, 100))
    y = square(square(square(x)))


@profile
def nested_square_3d_with_no_backprops():
    with Config.no_grad():
        x = Variable(np.random.rand(100, 100, 100))
        y = square(square(square(x)))


if __name__ == "__main__":
    nested_square_3d_with_no_backprops()
