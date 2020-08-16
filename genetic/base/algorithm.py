# ======================================================================================================================
#   Algoritmo Genetico - Principal
# ----------------------------------------------------------------------------------------------------------------------
#   Autor: Rafael Hideo Toyomoto
#   02/12/2018
# ======================================================================================================================


import time
from genetic.base.crossover import Crossover
from genetic.base.population import Population

# ======================================================================================================================


class Algorithm(object):

    def __init__(self, population, maxgen=100, mutation=0.05):
        """

        :param population:
        :type population: Population
        :param maxgen:
        :param mutation:
        """

        self.__population = population

        self.__generation = 0
        self.__maxgen = maxgen
        self.__stagnant = 0

        self.__mutation = mutation

    def run(self, debug=False):
        init_time = time.time()

        if debug:
            print('=' * 80)
            print("Initializing GA")

        # Primeira avaliacao
        self.__population.eval()

        if debug:
            print("time: %.3fs" % float(time.time() - init_time))
            print(str(self.__population))

        while not self.stop():
            run_time = time.time()

            # Reproduzir
            children = self.__population.children(n=1.0)

            # Crossover + Mutacao
            Crossover(children, mutation=self.__mutation)()

            # Reavaliar
            children.eval()

            # Juntar os melhores (evoluir)
            self.__population.evolve(children)

            # Contar uma nova geracao
            self.__generation += 1

            if debug:
                print('-' * 80)
                print("Generation: %d (time: %.3fs)" % (self.__generation, float(time.time() - run_time)))
                print(str(self.__population))

        return self.__population.best()

    def stop(self):
        if max(self.__population.elements(), key=lambda x: x.fitness()) == min(self.__population.elements(), key=lambda x: x.fitness()):
            if self.__stagnant == self.__maxgen/4:
                return True
            else:
                self.__stagnant += 1
        else:
            self.__stagnant = 0
        return self.__generation >= self.__maxgen

# ======================================================================================================================
