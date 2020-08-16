# ======================================================================================================================
#   Crossover
# ----------------------------------------------------------------------------------------------------------------------
#   Autor: Rafael Hideo Toyomoto
#   02/12/2018
# ======================================================================================================================


import random
from genetic.base.population import Population

# ======================================================================================================================


class Crossover(object):

    def __init__(self, population, swap=0.5, mutation=0.05):
        """
        Execucao do crossover e mutacao
        :param population:
        :type population: Population
        :param mutation:
        :type mutation: float
        """
        self.__population = population
        self.__swap_rate = swap
        self.__mutation_rate = mutation

    def __call__(self, *args, **kwargs):

        size = len(self.__population.elements()[0].genes if len(self.__population.elements()) > 0 else 0)

        n_changes = int(self.__swap_rate * size)
        pos_changes = [random.randrange(size) for i in range(n_changes)]

        for i in range(0, int(len(self.__population.elements())/2)*2, 2):
            father = self.__population.elements()[i]
            mother = self.__population.elements()[i+1]

            for j in range(0, len(father.genes)):
                if j in pos_changes:
                    father.genes[j].swap(mother.genes[j])
                if random.uniform(0, 1) <= self.__mutation_rate:
                    father.genes[j].shuffle()
                if random.uniform(0, 1) <= self.__mutation_rate:
                    mother.genes[j].shuffle()

# ======================================================================================================================
