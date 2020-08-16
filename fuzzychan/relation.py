# ==================================================================================================================== #
#   Author: Rafael Hideo Toyomoto                                                                                      #
#   Created: 19/09/2018                                                                                                #
# ==================================================================================================================== #

import math
from enum import Enum

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fuzzychan.base import FuzzySet


# ==================================================================================================================== #
#   EnumRelation
# -------------------------------------------------------------------------------------------------------------------- #
#   Enum de Normas implementadas
# ==================================================================================================================== #


class EnumRelation(Enum):
    tNorm1 = 't-norm1'
    tNorm4 = 't-norm4'
    tNorm9 = 't-norm9'
    sNorm4 = 's-norm4'
    sNorm9 = 's-norm9'
    Prod = 'prod'
    Min = 'min'
    Max = 'max'
    Sum = 'sum'


# ==================================================================================================================== #
#   Relation
# -------------------------------------------------------------------------------------------------------------------- #
#   Classe que representa relacoes fuzzy
# ==================================================================================================================== #


class FuzzyRelation(object):

    def __init__(self, kind=EnumRelation.Min, const=0.0, **kwargs):
        """
        Relacao Fuzzy
        :param kind: 
        :type kind: str|EnumRelation
        :param const: 
        :type const: int|float
        :param kwargs: 
        """""

        # Tipo que sera utilizado para relacionar os conjuntos
        self._kind = kind
        # Salvar uma constante que pode ser utilizada na operacao de relacionamento
        self._const = float(const)
        # Lista dos conjuntos fuzzy da relacao
        self._funcs = {}

        # Pegar os conjunto fuzzy passado no dicionario
        for var in kwargs:
            fuzzy_set = kwargs[var]
            if isinstance(fuzzy_set, FuzzySet):
                self._funcs[var] = fuzzy_set

    def __call__(self, *args, **kwargs):

        # Get 'kind'
        kind = self._kind

        # Get 'const'
        k = self._const

        # Computar as funcoes de pertinencia
        r_array = []
        for var in self._funcs:
            r_array.append(self._funcs[var].func(x=kwargs[var]))

        # Produtorio dos resultados
        r_prod = np.prod(r_array)

        # Somatorio dos resultados
        r_sum = np.sum(r_array)

        # Computar a relacao para a coordenada passada
        if kind == EnumRelation.tNorm1:
            return r_prod / (k + (1 - k) * (r_sum - r_prod))

        if kind == EnumRelation.tNorm4 or kind == EnumRelation.Prod:
            return r_prod

        if kind == EnumRelation.tNorm9 or kind == EnumRelation.Min:
            return min(r_array)

        if kind == EnumRelation.sNorm4 or kind == EnumRelation.Sum:
            return r_sum - r_prod

        if kind == EnumRelation.sNorm9 or kind == EnumRelation.Max:
            return max(r_array)

        return r_sum if len(self._funcs) == 1 else 0

    def matrix(self, complete=False):

        # Pegar os dominios de cada uma das funcoes
        domains = {}
        domains_size = []
        for var in self._funcs:
            domains[var] = self._funcs[var]._universe.domain.points
            domains_size.append(len(domains[var]))

        # Produto cartesiano
        prod_cart = np.meshgrid(*(domains.values()))

        new_domain = []
        # Expandir os arrays do produto cartesiado (cada eixos do resultado)
        for coord in prod_cart:
            new_domain.append(np.ravel(coord))

        results = []
        # Combinar os eixos expandidos (todas combinacoes possiveis de pontos)
        for values in zip(*new_domain):
            param = {}
            iter = 0

            # Criar o kwargs com as coordenadas do ponto atual
            for var in domains.keys():
                param[var] = values[iter]
                iter += 1

            # Calcular a relacao no ponto
            results.append(self(**param))

        # Reconstruir a matriz
        results = np.array(results).reshape(domains_size)

        if complete:
            prod_cart.append(results)
            return prod_cart
        return results

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
        if self.dimension() == 1:
            x, y = self.matrix(complete=True)
            axes = figure.add_subplot(111)
            axes.plot(x, y)

        # Plot 3D?
        elif self.dimension() >= 2:
            coord = self.matrix(complete=True)
            axes = figure.add_subplot(111, projection='3d')
            axes.plot_surface(coord[0], coord[1], coord[len(coord)-1])

        # Invalid
        else:
            return False

        axes.legend()
        return axes

    def alpha_cut(self, alpha, height=1):
        # TODO: tirar alfa corte da relacao (discreto)
        pass

    def support(self):
        # TODO: tirar o conjunto suporte (discreto/continuo)
        pass

    def dimension(self):
        return len(self._funcs)

# ==================================================================================================================== #
