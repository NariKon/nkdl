from core import Variable, Exp, Square, Add


def exp(x: Variable) -> Variable:
    return Exp()(x)


def square(x: Variable) -> Variable:
    return Square()(x)


def add(x0: Variable, x1: Variable) -> Variable:
    return Add()(x0, x1)
