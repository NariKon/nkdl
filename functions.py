from core import Variable, Exp, Square


def exp(x: Variable) -> Variable:
    return Exp()(x)


def square(x: Variable) -> Variable:
    return Square()(x)
