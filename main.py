import pandas as pd
import random
from sklearn.model_selection import train_test_split
from fuzzychan.function import MembershipFunc
from fuzzychan.base import FuzzyUniverse
from fuzzychan.classifier.wangmendel import WangMendelClassifier
from genetic.base.algorithm import Algorithm
from genetic.base.population import Population
from genetic.base.chromosome import Chromosome
from genetic.base.gene import Gene
from fuzzychan.inference.mamdani import MamdaniModel
from fuzzychan.inference.mamdani import EnumMamdaniDfzz
from fuzzychan.inference.mamdani import EnumMamdaniAggr
from fuzzychan.inference.mamdani import EnumMamdaniOper
from fuzzychan.inference.mamdani import EnumMamdaniImpl


def split_genes_to_fuzzy(gs):
    x1 = FuzzyUniverse("Variance", gs[0].min(), gs[0].max())
    x1["V_BAIXA"] = MembershipFunc(gs[0].min(), gs[0].value(), gs[1].value())
    x1["V_MEDIA"] = MembershipFunc(gs[2].value(), gs[3].value(), gs[4].value())
    x1["V_ALTA"] = MembershipFunc(gs[5].value(), gs[6].value(), gs[6].max())

    x2 = FuzzyUniverse("Skeness", gs[7].min(), gs[7].max())
    x2["S_BAIXA"] = MembershipFunc(gs[7].min(), gs[7].value(), gs[8].value())
    x2["S_MEDIA"] = MembershipFunc(gs[9].value(), gs[10].value(), gs[11].value())
    x2["S_ALTA"] = MembershipFunc(gs[12].value(), gs[13].value(), gs[13].max())

    x3 = FuzzyUniverse("Curtosis", gs[14].min(), gs[14].max())
    x3["C_BAIXA"] = MembershipFunc(gs[14].min(), gs[14].value(), gs[15].value())
    x3["C_MEDIA"] = MembershipFunc(gs[16].value(), gs[17].value(), gs[18].value())
    x3["C_ALTA"] = MembershipFunc(gs[19].value(), gs[20].value(), gs[20].max())

    x4 = FuzzyUniverse("Entropy", gs[21].min(), gs[21].max())
    x4["E_BAIXA"] = MembershipFunc(gs[21].min(), gs[21].value(), gs[22].value())
    x4["E_MEDIA"] = MembershipFunc(gs[23].value(), gs[24].value(), gs[25].value())
    x4["E_ALTA"] = MembershipFunc(gs[26].value(), gs[27].value(), gs[27].max())

    return x1, x2, x3, x4

def fitness_function(chromosome, data):
        """
        ::param chromosome:
        :type chromosome: Chromosome
        """

        gs = chromosome.genes

        if len(gs) != 28:
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

        if not (gs[14].value() < gs[15].value()) or not (gs[15].value() > gs[16].value()):
            return 0
        if not (gs[16].value() < gs[17].value() < gs[18].value()) or not (gs[18].value() > gs[19].value()):
            return 0
        if not (gs[19].value() < gs[20].value()):
            return 0
        if not (gs[14].value() < gs[15].value() < gs[20].value()):
            return 0

        if not (gs[21].value() < gs[22].value()) or not (gs[22].value() > gs[23].value()):
            return 0
        if not (gs[23].value() < gs[24].value() < gs[25].value()) or not (gs[25].value() > gs[26].value()):
            return 0
        if not (gs[26].value() < gs[27].value()):
            return 0
        if not (gs[21].value() < gs[24].value() < gs[27].value()):
            return 0

        x1, x2, x3, x4 = split_genes_to_fuzzy(gs)
        wmcls = WangMendelClassifier(variance=x1, skewness=x2, curtosis=x3, entropy=x4)
        wmcls.train(data.to_dict('records'), out_label='class', debug=False)
        fit = wmcls.get_fitness() * 100
        return fit

def create_first_genes(data):
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
    min = data["curtosis"].min()
    max = data["curtosis"].max()
    n_genes3 = [random.uniform(min, max) for i in range(0, 7)]
    n_genes3.sort()
    genes3 = [Gene(min, max, n) for n in n_genes3]
    genes3[1], genes3[2] = genes3[2], genes3[1]
    genes3[4], genes3[5] = genes3[5], genes3[4]

    """ Creating fourth universe """
    min = data["entropy"].min()
    max = data["entropy"].max()
    n_genes4 = [random.uniform(min, max) for i in range(0, 7)]
    n_genes4.sort()
    genes4 = [Gene(min, max, n) for n in n_genes4]
    genes4[1], genes4[2] = genes4[2], genes4[1]
    genes4[4], genes4[5] = genes4[5], genes4[4]

    return genes1, genes2, genes3, genes4

def func_y():
    f = FuzzyUniverse("Autenticidade", 0, 1)
    f["AAA"] = MembershipFunc(0, 0, 0.20)
    f["AA"] = MembershipFunc(0.20, 0.30, 0.40)
    f["AF"] = MembershipFunc(0.40, 0.50, 0.60)
    f["FF"] = MembershipFunc(0.60, 0.70, 0.80)
    f["FFF"] = MembershipFunc(0.80, 0.90, 1.00)
    return f

# Leitura dos dados de entrada.
DATA = pd.read_csv("dados_autent_bancaria.txt")
MAX_GEN = 1
MUTATION = 0.05

# Criação dos conjuntos de treinamento, validação e de testes.
array_train_validate, array_test = train_test_split(DATA.to_numpy(), test_size=0.2)
data_train_validate = pd.DataFrame(data=array_train_validate, columns=DATA.columns)
array_train, array_validate = train_test_split(data_train_validate.to_numpy(), test_size=0.2)
data_train = pd.DataFrame(data=array_train, columns=DATA.columns)
data_validate = pd.DataFrame(data=array_validate, columns=DATA.columns)
data_test = pd.DataFrame(data=array_test, columns=DATA.columns)

# Execução de um Algoritmo Genético para definir os limites de separação das classes.
# Criação dos primeiros genes do Algoritmo Genético.
genes1, genes2, genes3, genes4 = create_first_genes(data_train)
# Execução do Algoritmo Genético.
pop = Population(Chromosome(fitness_function, data_train, *(genes1 + genes2 + genes3 + genes4)), 50)
ga = Algorithm(pop, maxgen=MAX_GEN, mutation=MUTATION)
ga.run(debug=True)
# Seleção do melhor indivíduo.
best = pop.best()
# Recuperaçao dos genes com as separações.
gs = best.genes
x1, x2, x3, x4 = split_genes_to_fuzzy(gs)

# Aplicação do método de Wang Mendel para criar conjunto de regras.
wmcls = WangMendelClassifier(variance=x1, skewness=x2, curtosis=x3, entropy=x4)
wmcls.train(data_train.to_dict('records'), out_label='class', debug=False)
wmcls.print_status()

# Aplicação das regras Fuzzy selecionada para o conjunto de teste.

y = func_y()
mamdani = MamdaniModel(oper=EnumMamdaniOper.Prod,
                       impl=EnumMamdaniImpl.ConjPro,
                       aggr=EnumMamdaniAggr.Max,
                       dfzz=EnumMamdaniDfzz.MoM,
                       x1=x1,
                       x2=x2,
                       x3=x3,
                       x4=x4,
                       out=y)

mamdani.create_rule(x1='PS', x2='SM', out='MC')
mamdani.create_rule(x1='PS', x2='MM', out='M')
mamdani.create_rule(x1='PS', x2='GM', out='L')
mamdani.create_rule(x1='MS', x2='SM', out='C')
mamdani.create_rule(x1='MS', x2='MM', out='M')
mamdani.create_rule(x1='MS', x2='GM', out='L')
mamdani.create_rule(x1='GS', x2='SM', out='M')
mamdani.create_rule(x1='GS', x2='MM', out='L')
mamdani.create_rule(x1='GS', x2='GM', out='ML')

out = np.array([[int(i), int(j), mamdani(x1=i, x2=j)]
                for i in range(0, 110, 10) for j in range(0, 110, 10)])
np.savetxt("mamdani-mom.csv", out, fmt=['%d', '%d', '%.5f'], delimiter=";")
