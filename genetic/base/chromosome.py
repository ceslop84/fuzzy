# ======================================================================================================================
#   Cromossomo
# ----------------------------------------------------------------------------------------------------------------------
#   Autor: Rafael Hideo Toyomoto
#   02/12/2018
# ======================================================================================================================


import copy
from genetic.base.gene import Gene


# ======================================================================================================================


class Chromosome(object):

    def __init__(self, fitness_func, *args):
        self.genes = []
        self.__fitness = None
        self.__fitness_func = fitness_func

        # Coletar os genes passados como parametro (template)
        for gene in args:
            if not isinstance(gene, Gene):
                raise TypeError
            self.genes.append(gene)

    def __copy__(self):
        c = Chromosome(self.__fitness_func)
        c.genes = copy.deepcopy(self.genes)
        return c

    def regen(self):
        c = Chromosome(self.__fitness_func)
        for gene in self.genes:
            c.genes.append(copy.deepcopy(gene).shuffle())
        return c

    def __str__(self):
        return ("fitness=%.2f (" % float(0.0 if self.__fitness is None else self.__fitness)) + ", ".join(
            [str(gene) for gene in self.genes]) + ')'

    def fitness(self, force=False):
        if self.__fitness is None or force:
            self.__fitness = self.__fitness_func(self)
        return self.__fitness

# ======================================================================================================================
