# ==================================================================================================================== #
#   Author: Rafael Hideo Toyomoto                                                                                      #
#   Created: 18/08/2018                                                                                                #
# ==================================================================================================================== #

import abc
from fuzzychan.continuous.function import Function, Function2D


# ==================================================================================================================== #
#   Operator
# -------------------------------------------------------------------------------------------------------------------- #
#   Classe que representa uma operacao fuzzy (com funcoes 2D de pertinencia)
# ==================================================================================================================== #


class Operator(Function):
    __metaclass__ = abc.ABCMeta

    def __init__(self, color=None):
        Function.__init__(self, color=color)


# ==================================================================================================================== #
#   SingleOperator
# -------------------------------------------------------------------------------------------------------------------- #
#   Classe que representa uma operacao fuzzy com uma funcao
# ==================================================================================================================== #


class SingleOperator(Operator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, color=None):
        Operator.__init__(self, color=color)
        if not isinstance(func, Function):
            raise TypeError("SingleOperator: parameter func must be instance of Function.")
        self.function = func


# ----------------------------------------------------------------------------------------------------------------------


class Concentration(SingleOperator):
    def __init__(self, func, p, color=None):
        SingleOperator.__init__(self, func, color=color)
        p = float(p)
        self.p = p if p > 1 else (1 / p if p != 0 else 1)

    def calc(self, x):
        return self.function.calc(x) ** self.p

    def alpha_cut(self, alpha, height=1):
        alpha = alpha ** float(1/self.p)
        return self.function.alpha_cut(alpha, height)


# ----------------------------------------------------------------------------------------------------------------------


class Dilation(SingleOperator):
    def __init__(self, func, p, color=None):
        SingleOperator.__init__(self, func, color=color)
        p = float(p)
        self.p = p if 0 < p < 1 else (1 / p if p != 0 else 1)

    def calc(self, x):
        return self.function.calc(x) ** self.p

    def alpha_cut(self, alpha, height=1):
        alpha = alpha ** float(1/self.p)
        return self.function.alpha_cut(alpha, height)


# ----------------------------------------------------------------------------------------------------------------------


class Intensification(SingleOperator):
    def __init__(self, func, p, color=None):
        SingleOperator.__init__(self, func, color=color)
        self.p = p

    def calc(self, x):
        value = self.function.calc(x)
        if value <= 0.5:
            return (2 ** (self.p - 1)) * (value ** self.p)
        else:
            return 1 - (2 ** (self.p - 1)) * ((1 - value) ** self.p)


# ----------------------------------------------------------------------------------------------------------------------


class Fuzzification(SingleOperator):
    def calc(self, x):
        value = self.function.calc(x)
        if value <= 0.5:
            return (value / 2) ** 0.5
        else:
            return 1 - (((1 - value) / 2) ** 2)


# ----------------------------------------------------------------------------------------------------------------------


class Complement(SingleOperator):
    def calc(self, x):
        return 1 - self.function.calc(x)

    def alpha_cut(self, alpha, height=1):
        alpha = 1 - alpha
        return self.function.alpha_cut(alpha, height)


# ==================================================================================================================== #
#   MultipleOperator
# -------------------------------------------------------------------------------------------------------------------- #
#   Classe que representa uma operacao fuzzy com duas ou mais funcoes
# ==================================================================================================================== #


class MultipleOperator(Operator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func1, func2, color=None):
        Operator.__init__(self, color=color)
        self.functions = []
        self.add_func(func1)
        self.add_func(func2)

    def add_func(self, func):
        if not isinstance(func, Function):
            raise TypeError("MultipleOperator.add_func(): parameter func must be instance of Function.")
        self.functions.append(func)


# ----------------------------------------------------------------------------------------------------------------------


class Union(MultipleOperator):
    def calc(self, x):
        return max(func.calc(x) for func in self.functions)

    def alpha_cut(self, alpha, height=1):
        func1 = self.functions[0].alpha_cut(alpha, height)
        func2 = self.functions[1].alpha_cut(alpha, height)
        alpha_cut = Union(func1, func2)
        for i in range(2, len(self.functions)):
            alpha_cut.add_func(self.functions[i].alpha_cut(alpha, height))
        return alpha_cut

# ----------------------------------------------------------------------------------------------------------------------


class Intersection(MultipleOperator):
    def calc(self, x):
        return min(func.calc(x) for func in self.functions)

    def alpha_cut(self, alpha, height=1):
        func1 = self.functions[0].alpha_cut(alpha, height)
        func2 = self.functions[1].alpha_cut(alpha, height)
        alpha_cut = Intersection(func1, func2)
        for i in range(2, len(self.functions)):
            alpha_cut.add_func(self.functions[i].alpha_cut(alpha, height))
        return alpha_cut


# ==================================================================================================================== #
#   Operator2D
# -------------------------------------------------------------------------------------------------------------------- #
#   Classe que representa uma operacao fuzzy 2D (com funcoes 2D ou relacoes de pertinencia)
# ==================================================================================================================== #


class Operator2D(Function2D):
    __metaclass__ = abc.ABCMeta

    def __init__(self, color=None):
        Function2D.__init__(self, color=color)
        self.funcs = []

    def add_func(self, func):
        if not isinstance(func, Function2D):
            raise TypeError("RelationOperation.add_func(): parameter func must be instance of Function2D.")
        self.funcs.append(func)
        return self


# ----------------------------------------------------------------------------------------------------------------------


class Union2D(Operator2D):
    def __init__(self, func1, func2):
        Operator2D.__init__(self)
        self.add_func(func1)
        self.add_func(func2)

    def calc(self, x, y):
        return max([func.calc(x, y) for func in self.funcs])


# ----------------------------------------------------------------------------------------------------------------------


class Intersection2D(Operator2D):
    def __init__(self, func1, func2):
        Operator2D.__init__(self)
        self.add_func(func1)
        self.add_func(func2)

    def calc(self, x, y):
        return min([func.calc(x, y) for func in self.funcs])


# ==================================================================================================================== #
#   NormOperator
# -------------------------------------------------------------------------------------------------------------------- #
#   Classe que representa uma operacao fuzzy entre duas funcoes de pertinencia de dominios diferentes
# ==================================================================================================================== #


class NormOperator(Function2D):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func_x, func_y, color=None):
        Function2D.__init__(self, color=color)
        self.func_x = func_x
        self.func_y = func_y


# ----------------------------------------------------------------------------------------------------------------------


class TNormOperator(NormOperator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func_x, func_y, color=None):
        NormOperator.__init__(self, func_x, func_y, color=color)


# ----------------------------------------------------------------------------------------------------------------------


class TNorm1(TNormOperator):
    def __init__(self, func_x, func_y, pt):
        TNormOperator.__init__(self, func_x, func_y)
        self.pt = pt

    def calc(self, x, y):
        a = self.func_x.calc(x)
        b = self.func_y.calc(y)
        return a * b / (self.pt + (1 - self.pt) * (a + b - a * b))


# ----------------------------------------------------------------------------------------------------------------------


class TNorm9(TNormOperator):
    def calc(self, x, y):
        a = self.func_x.calc(x)
        b = self.func_y.calc(y)
        return min(a, b)


# ----------------------------------------------------------------------------------------------------------------------


class CoNormOperator(NormOperator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func_x, func_y, color=None):
        NormOperator.__init__(self, func_x, func_y, color=color)


# ----------------------------------------------------------------------------------------------------------------------


class CoNorm4(CoNormOperator):
    def calc(self, x, y):
        a = self.func_x.calc(x)
        b = self.func_y.calc(y)
        return a + b - a * b


# ----------------------------------------------------------------------------------------------------------------------


class CoNorm9(CoNormOperator):
    def calc(self, x, y):
        a = self.func_x.calc(x)
        b = self.func_y.calc(y)
        return max(a, b)

# ======================================================================================================================
