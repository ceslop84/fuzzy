# ======================================================================================================================
#   Gene
# ----------------------------------------------------------------------------------------------------------------------
#   Autor: Rafael Hideo Toyomoto
#   02/12/2018
# ======================================================================================================================


import random

# ======================================================================================================================


class Gene(object):

    def __init__(self, minv, maxv, value=None):
        self.__min = float(minv)
        self.__max = float(maxv)

        self.__value = random.uniform(self.__min, self.__max) if value is None else value

    def __copy__(self):
        return Gene(self.__min, self.__max, self.__value)

    def __str__(self):
        return "%.5f" % self.value()

    def swap(self, other):
        """
        Trocar informacao entre dois Genes
        :param other: Outro gene
        :type other: Gene
        :return: Proprio objeto
        """
        value = self.__value
        self.__value = other.__value
        other.__value = value
        return self

    def shuffle(self):
        #self.__value = random.uniform(self.__min, self.__max)
        self.__value = min(max(self.__value * (0.8 + random.uniform(0, 0.4)), self.__min), self.__max)
        return self

    def value(self):
        return self.__value

    def min(self):
        return self.__min

    def max(self):
        return self.__max
# ======================================================================================================================
