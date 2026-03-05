import numpy as np
from memory_profiler import profile
from core import Variable
from functions import square


@profile
def nested_square_loop(iter_max=10):
    for _ in range(iter_max):
        x = Variable(np.array(10_000))
        y = square(square(square(x)))


if __name__ == "__main__":
    nested_square_loop()
