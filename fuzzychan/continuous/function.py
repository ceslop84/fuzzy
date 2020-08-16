# ==================================================================================================================== #
#   Author: Rafael Hideo Toyomoto                                                                                      #
#   Created: 18/08/2018                                                                                                #
# ==================================================================================================================== #

import abc
import numpy as np


# ==================================================================================================================== #
#   Function
# -------------------------------------------------------------------------------------------------------------------- #
#   Classe que representa funcoes
# ==================================================================================================================== #


class Function(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, color=None):
        self.color = color

    @abc.abstractmethod
    def calc(self, x):
        pass

    @abc.abstractmethod
    def alpha_cut(self, alpha, height=1):
        pass

    def gen(self, interval):
        return [self.calc(x) for x in interval]


class Function2D(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, color=None):
        self.color = color

    @abc.abstractmethod
    def calc(self, x, y):
        pass

    def gen(self, interval_x, interval_y):
        (X, Y) = np.meshgrid(interval_x, interval_y)
        z = np.array([self.calc(x, y) for (x, y) in zip(np.ravel(X), np.ravel(Y))])
        return X, Y, z.reshape(X.shape)


# ==================================================================================================================== #
#   MembershipFunction
# -------------------------------------------------------------------------------------------------------------------- #
#   Classe que representa funcoes de pertinencia (fuzzy)
# ==================================================================================================================== #


class MembershipFunction(Function):
    def __init__(self, height=1):
        Function.__init__(self)
        self.height = height

    @abc.abstractmethod
    def print_support(self):
        pass

    @abc.abstractmethod
    def print_core(self):
        pass

    @abc.abstractmethod
    def inv_calc(self, y):
        return None, None

    @abc.abstractmethod
    def card(self):
        pass

    def alpha_cut(self, alpha, height=1):
        p1, p2 = self.inv_calc(alpha)
        alpha_cut = RectangularMFunction(p1, p2)
        alpha_cut.set_height(height)
        return alpha_cut

    def inclusion(self, target, interval):
        card, sum = 0, 0
        for x in interval:
            card += self.calc(x)
            sum += max(0, (self.calc(x) - target.calc(x)))
        if card == 0:
            return 0
        return (card - sum) / card

    def get_height(self):
        return self.height

    def set_height(self, height):
        self.height = height
        return self


# ==================================================================================================================== #
#   TrapezoidalMFunction
# -------------------------------------------------------------------------------------------------------------------- #
#   Classe que representa funcoes de pertinencia trapezoidal
# ==================================================================================================================== #


class TrapezoidalMFunction(MembershipFunction):
    def __init__(self, a, m, n, b, height=1):
        MembershipFunction.__init__(self, height=height)
        self.a = a
        self.m = m
        self.n = n
        self.b = b

    def print_support(self):
        return str("(%.4f, %.4f)" % (self.a, self.b))

    def print_core(self):
        return str("[%.4f, %.4f]" % (self.m, self.n))

    def calc(self, x):
        if self.a < x < self.m:
            return self.height * (x - self.a) / (self.m - self.a)
        elif self.m <= x <= self.n:
            return self.height
        elif self.n < x < self.b:
            return self.height * (self.b - x) / (self.b - self.n)
        return 0

    def inv_calc(self, y):
        if 0 > y > 1:
            return None, None
        elif y == 1:
            return self.m, self.n
        return float(y * float(self.m - self.a) / self.height + self.a), float(
            self.b - y * float(self.b - self.n) / self.height)

    def card(self):
        return self.height * ((self.m - self.a) / 2 + (self.n - self.m) + (self.b - self.n) / 2)


# ==================================================================================================================== #
#   TriagularMFunction
# -------------------------------------------------------------------------------------------------------------------- #
#   Classe que representa funcoes de pertinencia triangular
# ==================================================================================================================== #


class TriagularMFunction(TrapezoidalMFunction):
    def __init__(self, a, m, b, height=1):
        TrapezoidalMFunction.__init__(self, a, m, m, b, height=height)


# ==================================================================================================================== #
#   RectangularMFunction
# -------------------------------------------------------------------------------------------------------------------- #
#   Classe que representa funcoes de pertinencia retangular
# ==================================================================================================================== #


class RectangularMFunction(TrapezoidalMFunction):
    def __init__(self, a, b, height=1):
        TrapezoidalMFunction.__init__(self, a, a, b, b, height=height)

# ==================================================================================================================== #
