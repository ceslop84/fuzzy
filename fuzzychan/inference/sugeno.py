# ======================================================================================================================
#   Modelo Sugeno de inferencia Fuzzy
# ----------------------------------------------------------------------------------------------------------------------
#   Autor: Rafael Hideo Toyomoto
#   25/10/2018
# ======================================================================================================================


from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Dict

from fuzzychan.base import FuzzyUniverse, MembershipFunc, FuzzySet, Domain
from fuzzychan.relation import FuzzyRelation
from fuzzychan.rule import fuzzy_rule_compute
from fuzzychan.operator import FuzzyOperator

from fuzzychan.rule import EnumRule as EnumSugenoImpl
from fuzzychan.relation import EnumRelation as EnumSugenoOper
from fuzzychan.relation import EnumRelation as EnumSugenoAggr
from fuzzychan.operator import EnumFuzzyOper


# ==================================================================================================================== #


class EnumSugenoDfzz(Enum):
    Avg = 'Avg'
    Sum = 'Sum'


# ==================================================================================================================== #
#   SugenoRule
# -------------------------------------------------------------------------------------------------------------------- #
#   Regra do Modelo Takagi-Sugeno
# ==================================================================================================================== #


class SugenoRule(object):
    def __init__(self, antecedent, consequent, *args, **kwargs):
        """
        Regra Fuzzy
        :param antecedent:
        :type antecedent: FuzzyRelation
        :param consequent:
        :type consequent: Dict[str, List[Union[int, float]]]
        :param args:
        :param kwargs:
        """

        # Salvar o antecedente e consequente
        self.__antecedent = antecedent
        self.__consequent = consequent

    def getAntecedent(self):
        return self.__antecedent

    def getConsequent(self):
        return self.__consequent

    def __call__(self, **kwargs):

        # Verificar se tem input suficiente
        inp_size = self.__antecedent.dimension()
        if len(kwargs) != inp_size:
            raise AttributeError

        # Computar o antecedente (peso)
        weight = self.__antecedent(**kwargs)

        # Computar f(antecedente) ou consequente
        total = 0
        for var in self.__consequent:

            # Elemento constante
            if var == 'const':
                total += self.__consequent[var]
            # Funcao de algum antecedente
            else:
                func = self.__consequent[var]
                x = float(kwargs[var])
                xi = x
                for coef in reversed(func):
                    total += coef * xi
                    xi *= x

        # Return: [wi, f(x1,x2,...,xn)]
        return [weight, total]


# ==================================================================================================================== #
#   SugenoModel
# -------------------------------------------------------------------------------------------------------------------- #
#   Modelo Sugeno para inferencia
# ==================================================================================================================== #


class SugenoModel(object):

    def __init__(self,
                 oper=EnumSugenoOper.Min,
                 dfzz=EnumSugenoDfzz.Avg,
                 out=None,
                 **kwargs):
        """
        Modelo Sugeno de inferencia Fuzzy
        :param oper: AND ou OR + semantica do operador de agregacao do antecedente. Matlab: AND or OR method
        :type oper: EnumSugenoOper
        :param dfzz: Metodo de Defuzzificacao. Matlab: Defuzzification
        :type dfzz: EnumSugenoDfzz
        :param out: passar o output (Y)
        :type out: FuzzyUniverse
        :param kwargs: passar os inputs (Xi)
        """

        self.__input = {}
        self.__output = out
        if out is None:
            raise AttributeError

        # Processar todos universos adicionados a esse modelo
        for var in kwargs:
            # Apenas universos fuzzy sao aceitos como um input/output
            if not isinstance(kwargs[var], FuzzyUniverse):
                raise TypeError
            # Demais serao apenas um input
            self.__input[var] = kwargs[var]

        self.__oper = oper
        self.__dfzz = dfzz

        # Criar um banco de regras para esse modelo
        self.__rules = []

    def create_rule(self, out=None, **kwargs):
        """
        Criar uma regra para esse modelo
        :param out:
        :type out: Union[Dict[str, List[Union[int, float]]], int, float]
        :param kwargs:
        :return:
        """

        # Precisamos de um label de  selecao para cada input
        if len(kwargs) != len(self.__input):
            raise AttributeError

        # Criar o Antecedente da regra
        funcs = {}
        for var in kwargs:
            # Preparar os argumentos para passar para criacao da Relacao Fuzzy que representara o antecedente
            label = str(kwargs[var])
            funcs[var] = self.__input[var][label]
        antecedent = FuzzyRelation(kind=self.__oper, **funcs)

        # Criar o Consequente da regra
        consequent = out if isinstance(out, dict) else {'const': float(out)}

        rule = SugenoRule(antecedent, consequent)
        self.__rules.append(rule)

    def __call__(self, *args, **kwargs):

        # Eh necessario um valor para cada input
        if len(kwargs) != len(self.__input):
            raise AttributeError

        # Soma dos produtos parciais dos pesos com o resultado da funcao y
        sum_func = 0
        # Soma dos pesos
        sum_weight = 0
        for rule in self.__rules:  # type: SugenoRule
            # Computar a regra
            weight_i, func_i = rule(**kwargs)

            sum_func += weight_i * func_i
            sum_weight += weight_i

        return sum_func / sum_weight

# ==================================================================================================================== #
