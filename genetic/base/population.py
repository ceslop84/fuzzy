# ======================================================================================================================
#   Conjunto de Cromossomos
# ----------------------------------------------------------------------------------------------------------------------
#   Autor: Rafael Hideo Toyomoto
#   02/12/2018
# ======================================================================================================================


import copy
from typing import List, Any
from genetic.base.chromosome import Chromosome

# ======================================================================================================================


class Population(object):

    def __init__(self, chromosome, size):
        self.__items = []

        for i in range(max(1, size)):
            self.__items.append(chromosome.regen())

    def elements(self):
        """

        :return:
        :rtype: List[Chromosome]
        """
        return self.__items

    def children(self, n=0.5):
        """
        Criar uma populacao nova de filhos dos cromossomos dessa populacao
        :param n: Se int, tamanho bruto. Se float (0 a 1), percentual
        :return: Nova populacao filha
        """

        if isinstance(n, float) and 0 <= n <= 1:
            n = int(len(self.__items) * n)
        else:
            n = int(max(min(n, len(self.__items)), 0))

        p = Population(self.__items[0], len(self.__items))
        p.__items = copy.deepcopy(self.__items[:n])

        return p

    def eval(self):
        """
        Avaliar a populacao toda e retornar o somatorio dos fitness
        :return: Fitness da populacao
        """
        fitness = 0
        for chromosome in self.__items:
            fitness += float(chromosome.fitness(force=True))
        return fitness

    def evolve(self, children):
        original_size = len(self.__items)
        self.__items += children.__items
        if len(self.__items) > 0:
            self.__items.sort(key=lambda chromosome: chromosome.fitness(), reverse=True)
            self.__items = self.__items[:original_size]

    def best(self):
        if len(self.__items) > 0:
            self.__items.sort(key=lambda chromosome: chromosome.fitness(), reverse=True)
            return self.__items[0]
        return None

    def __str__(self):
        return '\n'.join(
            [('[' + str(self.__items.index(chromosome)) + "] \t" + str(chromosome)) for chromosome in self.__items])

# ======================================================================================================================
