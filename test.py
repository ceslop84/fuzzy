import pandas as pd
from fuzzychan.function import MembershipFunc
from fuzzychan.base import FuzzyUniverse
from fuzzychan.classifier.wangmendel import WangMendelClassifier
from sklearn.model_selection import train_test_split
import random
from genetic.base.algorithm import Algorithm as GaAlgorithm
from genetic.base.population import Population as GaPopulation
from genetic.base.chromosome import Chromosome
from genetic.base.gene import Gene


def generate_universes(gs, sample):
    x1 = FuzzyUniverse("Attr1", gs[0].min(), gs[0].max(), sample)
    x1["A1"] = MembershipFunc(gs[0].min(), gs[0].value(), gs[1].value())
    x1["A2"] = MembershipFunc(gs[2].value(), gs[3].value(), gs[4].value())
    x1["A3"] = MembershipFunc(gs[5].value(), gs[6].value(), gs[0].max())

    x2 = FuzzyUniverse("Attr2", gs[7].min(), gs[7].max(), sample)
    x2["B1"] = MembershipFunc(gs[7].min(), gs[7].value(), gs[8].value())
    x2["B2"] = MembershipFunc(gs[9].value(), gs[10].value(), gs[11].value())
    x2["B3"] = MembershipFunc(gs[12].value(), gs[13].value(), gs[7].max())

    return x1, x2

def fitness_function(chromosome):
        """
        ::param chromosome:
        :type chromosome: Chromosome
        """

        gs = chromosome.genes

        if len(gs) != 14:
            return 0

        if not (gs[0].value() < gs[1].value()) or not (gs[1].value() > gs[2].value()):
            return 0
        if not (gs[2].value() < gs[3].value() < gs[4].value()) or not (gs[4].value() > gs[5].value()):
            return 0
        if not (gs[5].value() < gs[6].value()):
            return 0
        if not (gs[0].value() < gs[3].value() < gs[6].value()):
            return 0

        if not (gs[7].value() < gs[8].value()) or not (gs[8].value() > gs[9].value()):
            return 0
        if not (gs[9].value() < gs[10].value() < gs[11].value()) or not (gs[11].value() > gs[12].value()):
            return 0
        if not (gs[12].value() < gs[13].value()):
            return 0
        if not (gs[7].value() < gs[10].value() < gs[13].value()):
            return 0

        x1, x2 = generate_universes(gs, sample)

        wmcls = WangMendelClassifier(variance=x1, skewness=x2)
        d = data.to_dict('records')

        wmcls.train(d, out_label='class', debug=False)
        fit = wmcls.get_fitness() * 100
        return fit

# Leitura dos dados de entrada.
data = pd.read_csv("dados_autent_bancaria.txt")
sample = 1000

# Criação dos conjuntos de treinamento e de testes.
# x = data.to_numpy()
# y = data['class'].to_numpy()
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

""" Creating first universe """
min = data["variance"].min()
max = data["variance"].max()
n_genes1 = [random.uniform(min, max) for i in range(0, 7)]
n_genes1.sort()
genes1 = [Gene(min, max, n) for n in n_genes1]
genes1[1], genes1[2] = genes1[2], genes1[1]
genes1[4], genes1[5] = genes1[5], genes1[4]

""" Creating second universe """
min = data["skewness"].min()
max = data["skewness"].max()
n_genes2 = [random.uniform(min, max) for i in range(0, 7)]
n_genes2.sort()
genes2 = [Gene(min, max, n) for n in n_genes2]
genes2[1], genes2[2] = genes2[2], genes2[1]
genes2[4], genes2[5] = genes2[5], genes2[4]

""" Creating third universe """
min = data["skewness"].min()
max = data["skewness"].max()
n_genes2 = [random.uniform(min, max) for i in range(0, 7)]
n_genes2.sort()
genes2 = [Gene(min, max, n) for n in n_genes2]
genes2[1], genes2[2] = genes2[2], genes2[1]
genes2[4], genes2[5] = genes2[5], genes2[4]

""" Genetic Algorithm Tuning """
pop = GaPopulation(Chromosome(fitness_function, *(genes1 + genes2)), 50)
ga = GaAlgorithm(pop, maxgen=100, mutation=0.15)
ga.run(debug=True)
print()
# best = pop.best()
# gs = best.genes

# x1 = FuzzyUniverse("Attr1", 0, 100, 1000)
# x1["A1"] = MembershipFunc(0, 15, 33)
# x1["A2"] = MembershipFunc(34, 45, 66)
# x1["A3"] = MembershipFunc(67, 75, 100)
#
# x2 = FuzzyUniverse("Attr2", 0, 100, 1000)
# x2["B1"] = MembershipFunc(0, 15, 33)
# x2["B2"] = MembershipFunc(34, 45, 66)
# x2["B3"] = MembershipFunc(67, 75, 100)
#
# """ WangMendel Classifier to show results """
# wmcls = WangMendelClassifier(x1=x1, x2=x2)
# wmcls.train(df.to_dict('records'), out_label='cls', debug=False)
# wmcls.print_status()
