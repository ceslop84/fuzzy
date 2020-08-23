# ==================================================================================================================== #
#   Author: Rafael Hideo Toyomoto                                                                                      #
#   Created: 19/09/2018                                                                                                #
# ==================================================================================================================== #

import math
from enum import Enum

import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fuzzychan.base import FuzzySet
from fuzzychan.relation import FuzzyRelation


# ==================================================================================================================== #


def fuzzy_rule_compute(a, b, kind):
    """
    Computar a regra para a coordenada passada
    :param a: Valor do antecedente
    :type a: int|float
    :param b: Valor do consequente
    :type b: int|float
    :param kind: Semantica da regra
    :type kind: str|EnumRule
    :return: int|float
    """
    if kind == EnumRule.ConjMin:
        return min(a, b)
    if kind == EnumRule.ConjPro:
        return a * b
    if kind == EnumRule.DisjMax:
        return max(a, b)
    if kind == EnumRule.ImplLSWMin:
        return min(1, min(1 - a, b))
    if kind == EnumRule.ImplLSWPro:
        return min(1, (1 - a) * b)
    if kind == EnumRule.ImplGodel:
        return 1 if a <= b else b
    if kind == EnumRule.ImplKleene:
        return max(1 - a, b)
    if kind == EnumRule.ImplZadeh:
        return max(1 - a, min(a, b))
    return 0


def fuzzy_conclusion(rule, test):
    """
    Gerar conclusao a partir de uma regra e uma entrada linguistica
    :param rule: Regra Fuzzy
    :type rule: FuzzyRule|ndarray
    :param test: Relacao de entrada
    :type test: FuzzyRelation|ndarray
    :return: ndarray
    """

    # Discretizar a regra e a entrada
    # FIXME: verificar se B nao tem funcoes de dominio nao presente na regra
    a = rule.matrix() if isinstance(rule, FuzzyRule) else rule
    b = test.matrix() if isinstance(test, FuzzyRelation) else test

    # Diferenca de dimensao (Naturalmente, dimensao de A pode ser maior que B)
    # Se dimencao de B for maior, basta ignorar variaveis sem utilidade de B
    diff = a.ndim - b.ndim
    if diff > 0:

        # Extensao Cilindrica da entrada
        sample = len(a)
        fix = [sample if diff > iter else 1 for iter in range(0, a.ndim)]
        b = np.tile(b, fix)
    
    results = []
    for ai, bi in zip(np.ravel(a), np.ravel(b)):

        # Norma-T qualquer (escolhida -> min)
        r = min(ai, bi)
        results.append(r)

    return np.array(results).reshape(a.shape)


# ==================================================================================================================== #
#   EnumRelation
# -------------------------------------------------------------------------------------------------------------------- #
#   Enum de Normas implementadas
# ==================================================================================================================== #


class EnumRule(Enum):
    ConjMin = 'conj-min'
    ConjPro = 'conj-pro'
    DisjMax = 'disj'
    ImplLSWMin = 'impl-lucasiewicz-min'
    ImplLSWPro = 'impl-lucasiewicz-pro'
    ImplGodel = 'impl-godel'
    ImplKleene = 'impl-kleene'
    ImplZadeh = 'impl-zadeh'


# ==================================================================================================================== #
#   Relation
# -------------------------------------------------------------------------------------------------------------------- #
#   Classe que representa relacoes fuzzy
# ==================================================================================================================== #


class FuzzyRule(object):

    def __init__(self, antecedent, consequent, kind=EnumRule.ConjMin, *args, **kwargs):
        """
        Regra Fuzzy
        :param antecedent:
        :type antecedent: FuzzyRelation
        :param consequent:
        :type consequent: FuzzyRelation
        :param kind:
        :type kind: str|EnumRule
        :param args:
        :param kwargs:
        """

        # Tipo que sera utilizado para relacionar os conjuntos
        self._kind = kind
        # Lista dos conjuntos fuzzy da relacao
        self._antecedent = antecedent
        self._consequent = consequent

    def __call__(self, *args, **kwargs):

        # Get 'kind'
        kind = self._kind

        # Get values of antecedent and consequent
        a = args[0] if len(args) > 1 else self._antecedent(*args, **kwargs)
        b = args[1] if len(args) > 1 else self._consequent(*args, **kwargs)

        # Computar a regra para a coordenada passada
        return fuzzy_rule_compute(a, b, kind)

    def matrix(self, complete=False):

        # Gerar as matrizes para regra
        a = self._antecedent.matrix()
        b = self._consequent.matrix()

        # Produto cartesiano entre os dominios das funcoes do antecedente e do consequente
        domains = [self._antecedent._funcs[key]._universe.domain.points for key in self._antecedent._funcs]
        for key in self._consequent._funcs:
            func =  self._consequent._funcs[key]
            domains.append(func._universe.domain.points)
        domain = np.meshgrid(*domains)

        # Produto cartesiano entre as matrizes de resultados (somar as dimensoes)
        A, B = np.meshgrid(a, b)
        results = np.array([self(val_a, val_b) for (val_a, val_b) in zip(np.ravel(A), np.ravel(B))]).reshape([len(points) for points in domain])
        
        if complete:
            domain.append(results)
            return domain
        return results

    def conclusion(self, test):
        """
        Gerar uma conclusao para essa regra
        :param input: Entrada
        :type input: FuzzyRelation
        """
        # TODO: suporte a dois modelos diferentes
        fuzzy_conclusion(self, test)


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

        # Plot 2D?
        if len(self._funcs) == 1:
            x, y = self.matrix(complete=True)
            axes = figure.add_subplot(111)
            axes.plot(x, y)

        # Plot 3D?
        elif len(self._funcs) >= 2:
            coord = self.matrix(complete=True)
            axes = figure.add_subplot(111, projection='3d')
            axes.plot_surface(coord[0], coord[1], coord[len(coord) - 1])

        # Invalid
        else:
            return False

        axes.legend()
        return axes

    def alpha_cut(self, alpha, height=1):

        def explore_array(param):
            if isinstance(param, list):
                if isinstance(param[0], list):
                    return [explore_array(param)]
                else:
                    return [explore_array(item) for item in param]
            else:
                return param

        # TODO: tirar alfa corte da relacao (discreto)
        pass

    def support(self):
        # TODO: tirar o conjunto suporte (discreto/continuo)
        pass

# ==================================================================================================================== #
