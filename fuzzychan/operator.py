# ==================================================================================================================== #
#   Author: Rafael Hideo Toyomoto                                                                                      #
#   Created: 18/08/2018                                                                                                #
# ==================================================================================================================== #

from enum import Enum

from fuzzychan.function import MembershipFunc


# ==================================================================================================================== #
#   Operacoes em conjuntos fuzzy
# ==================================================================================================================== #

# ==================================================================================================================== #
#   EnumRelation
# -------------------------------------------------------------------------------------------------------------------- #
#   Enum de Normas implementadas
# ==================================================================================================================== #


class EnumFuzzyOper(Enum):
    # One Term
    Concentration = 'conc'
    Dilation = 'dilt'
    Intensification = 'ints'
    Fuzzification = 'fuzz'
    Complement = 'comp'
    # Two Term
    Union = 'union'
    Intersection = 'inter'


# ==================================================================================================================== #
#   EnumRelation
# -------------------------------------------------------------------------------------------------------------------- #
#   Enum de Normas implementadas
# ==================================================================================================================== #


class FuzzyOperator(MembershipFunc):

    def __init__(self, *args, **kwargs):
        #super(FuzzyOperator, self).__init__(args, kwargs)
        self._height = float(kwargs['height']) if 'height' in kwargs.keys() else 1.0

        # Get 'kind'
        self._kind = kind = kwargs['kind']

        # One Term
        if (kind == EnumFuzzyOper.Concentration or
                kind == EnumFuzzyOper.Dilation or
                kind == EnumFuzzyOper.Intensification or
                kind == EnumFuzzyOper.Fuzzification or
                kind == EnumFuzzyOper.Complement):
            self._func = args[0]
            assert isinstance(self._func, MembershipFunc)
        else:
            self._func = [args[0], args[1]]

        if (kind == EnumFuzzyOper.Concentration or
                kind == EnumFuzzyOper.Dilation or
                kind == EnumFuzzyOper.Intensification):
            self._p = int(kwargs['p']) if 'p' in kwargs.keys() else 2

    # Two Term

    def calc(self, x):
        # Get 'kind'
        kind = self._kind

        # === One Term ===

        if kind == EnumFuzzyOper.Concentration:
            return (self._func(x=x)) ** self._p
        if kind == EnumFuzzyOper.Dilation:
            return (self._func(x=x)) ** (1/float(self._p))
        if kind == EnumFuzzyOper.Intensification:
            value = self._func(x=x)
            if value <= 0.5:
                return (2 ** (self._p - 1)) * (value ** self._p)
            else:
                return 1 - (2 ** (self._p - 1)) * ((1 - value) ** self._p)
        if kind == EnumFuzzyOper.Fuzzification:
            value = self._func(x=x)
            if value <= 0.5:
                return (value / 2) ** 0.5
            else:
                return 1 - (((1 - value) / 2) ** 2)
        if kind == EnumFuzzyOper.Complement:
            return 1 - self._func(x=x)

        # === Two Term ===

        if kind == EnumFuzzyOper.Union:
            return max(self._func[0](x=x), self._func[1](x=x))
        if kind == EnumFuzzyOper.Intersection:
            return min(self._func[0](x=x), self._func[1](x=x))

        return 0

    def inv(self, mi):
        pass

    def alpha_cut(self, alpha, height=1):
        # FIXME: valido apenas para alguns tipos

        # Calculo dos pontos inversos
        p1, p2 = self(mi=alpha, inv='true')

        # Criar uma funcao crisp para representar o alfa corte
        return MembershipFunc(p1, p2, kind='crisp', height=height)

    def support(self):
        # FIXME: valido apenas para alguns tipos
        return MembershipFunc(self._a, self._b, kind='crisp')

    def core(self):
        # FIXME: valido apenas para alguns tipos
        return MembershipFunc(self._m, self._n, kind='crisp')

# ==================================================================================================================== #
