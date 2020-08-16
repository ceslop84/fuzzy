# ==================================================================================================================== #
#   Author: Rafael Hideo Toyomoto                                                                                      #
#   Created: 15/09/2018                                                                                                #
# ==================================================================================================================== #

from numpy import linspace

from matplotlib import pyplot as plt
from matplotlib.pyplot import Figure

from fuzzychan.function import MembershipFunc
from fuzzychan.operator import EnumFuzzyOper, FuzzyOperator


# ==================================================================================================================== #
#   Domain
# -------------------------------------------------------------------------------------------------------------------- #
#   Classe que representa um dominio com ponto inicial, ponto final e taxa de amostragem (discretizacao)
# ==================================================================================================================== #


class Domain(object):

    def __init__(self, label="", xi=0.0, xf=10.0, sample=100):
        """
        Dominio de um universo Fuzzy
        :param label: Nome do universo
        :type label: str
        :param xi: ponto inicial
        :type xi: int|float
        :param xf: ponto final
        :type xf: int|float
        :param sample: taxa de discretizacao
        :type sample: int
        """
        self.id = label + "-" + str(xi) + "-" + str(xf) + "-" + str(sample)
        self.points = linspace(float(xi), float(xf), int(sample))
        self.limits = (float(xi), float(xf))

    def __eq__(self, other):
        if isinstance(other, Domain):
            return self.id == other.id
        return False

    def __iter__(self):
        for x in self.points:
            yield x


# ==================================================================================================================== #
#   FuzzyUniverse
# -------------------------------------------------------------------------------------------------------------------- #
#   Classe que representa um universo fuzzy
# ==================================================================================================================== #


class FuzzyUniverse(dict):

    def __init__(self, label, xi, xf, sample, **kwargs):
        """
        Universo Fuzzy (discretizado)
        :param label: Nome do universo
        :type label: str
        :param xi: ponto inicial
        :type xi: int|float
        :param xf: ponto final
        :type xf: int|float
        :param sample: taxa de discretizacao
        :type sample: int
        :param kwargs: Criar conjuntos fuzzy no universo
        """
        super(FuzzyUniverse, self).__init__(**kwargs)

        # salvar o nome
        self.name = str(label)

        # criar o dominio dessa variavel
        self.domain = Domain(label=label, xi=xi, xf=xf, sample=sample)

    def __setitem__(self, key, value):
        """
        Criar um termo nesse universo
        :param key:
        :type key: str
        :param value:
        :type value: MembershipFunc|FuzzySet
        :return:
        """
        if isinstance(value, MembershipFunc):
            value = FuzzySet(self, func=value)
        if not isinstance(value, FuzzySet):
            raise TypeError("FuzzyUniverse dict only accept instance of 'FuzzySet'")
        super(FuzzyUniverse, self).__setitem__(key, value)

    def __getitem__(self, item):
        """

        :param item:
        :type item: str
        :return: FuzzySet
        """

        terms = item.split(' ')

        fz_set = super(FuzzyUniverse, self).__getitem__(terms[len(terms)-1])

        if len(terms) == 1:
            return fz_set

        func = fz_set.func

        for term_i in reversed(range(len(terms)-1)):
            if terms[term_i] == 'muito':
                func = FuzzyOperator(func, kind=EnumFuzzyOper.Concentration, p=2)
            elif terms[term_i] == 'nao':
                func = FuzzyOperator(func, kind=EnumFuzzyOper.Complement)
            elif terms[term_i] == 'levemente':
                func2 = FuzzyOperator(func, kind=EnumFuzzyOper.Concentration, p=2)
                func2 = FuzzyOperator(func2, kind=EnumFuzzyOper.Complement)
                func = FuzzyOperator(func, func2, kind=EnumFuzzyOper.Intersection)

        return FuzzySet(self, func=func)

    def plot(self, vars=None, figure=None):
        """
        Plotar o universo pela biblioteca 'matplotlib'
        :param vars: Filtro do plot
        :type vars: str|list
        :param figure:
        :type figure: Figure
        :return:
        """
        if figure is None:
            figure = plt.figure()

        axes = figure.add_subplot(111)

        for label in self.__iter__():
            # filter? (vars can be str or list)
            if vars is None or (isinstance(vars, str) and label == vars) or (isinstance(vars, list) and label in vars):
                axes.plot(self.domain.points, self[label].gen(), label=label)

        axes.legend()
        return axes


# ==================================================================================================================== #
#   FuzzySet
# -------------------------------------------------------------------------------------------------------------------- #
#   Classe que representa um conjunto fuzzy
# ==================================================================================================================== #


class FuzzySet(object):

    def __init__(self, universe, *args, **kwargs):
        """
        Conjunto Fuzzy
        :param universe: Universo desse conjunto
        :type universe: FuzzyUniverse
        :param args:
        :param kwargs:
        """
        if not isinstance(universe, FuzzyUniverse):
            raise TypeError("FuzzySet: parameter 'universe' must be instance of 'FuzzyUniverse'")
        self._universe = universe

        if 'func' in kwargs.keys():
            self.func = kwargs['func']
        else:
            self.func = MembershipFunc(*args, **kwargs)

    def gen(self):
        return [self.func(x=x) for x in self._universe.domain]

# ==================================================================================================================== #
