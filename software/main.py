"""Módulo principal para calcular a autencidade de uma nota através de técnicas Fuzzy."""
import os
import random
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from genetic.base.algorithm import Algorithm
from genetic.base.population import Population
from genetic.base.chromosome import Chromosome
from genetic.base.gene import Gene
from fuzzychan.function import MembershipFunc
from fuzzychan.base import FuzzyUniverse
from fuzzychan.classifier.wangmendel import WangMendelClassifier
from fuzzychan.inference.mamdani import MamdaniModel
from fuzzychan.inference.mamdani import EnumMamdaniDfzz
from fuzzychan.inference.mamdani import EnumMamdaniAggr
from fuzzychan.inference.mamdani import EnumMamdaniOper
from fuzzychan.inference.mamdani import EnumMamdaniImpl


def split_genes_to_fuzzy(genes):
    """ Método para separar um gene de tamanho 28 elementos em 4 de 7 elementos.

    Params:
        genes (gene): gene completo que será separado em múltiplos genes.

    Returns:
        Gene: Objeto do tipo Gene.
        Gene: Objeto do tipo Gene.
        Gene: Objeto do tipo Gene.
        Gene: Objeto do tipo Gene.
    """
    var_g = FuzzyUniverse("Variance", genes[0].min(), genes[0].max())
    var_g["V_BAIXA"] = MembershipFunc(genes[0].min(), genes[0].value(), genes[1].value())
    var_g["V_MEDIA"] = MembershipFunc(genes[2].value(), genes[3].value(), genes[4].value())
    var_g["V_ALTA"] = MembershipFunc(genes[5].value(), genes[6].value(), genes[6].max())

    skew_g = FuzzyUniverse("Skeness", genes[7].min(), genes[7].max())
    skew_g["S_BAIXA"] = MembershipFunc(genes[7].min(), genes[7].value(), genes[8].value())
    skew_g["S_MEDIA"] = MembershipFunc(genes[9].value(), genes[10].value(), genes[11].value())
    skew_g["S_ALTA"] = MembershipFunc(genes[12].value(), genes[13].value(), genes[13].max())

    curt_g = FuzzyUniverse("Curtosis", genes[14].min(), genes[14].max())
    curt_g["C_BAIXA"] = MembershipFunc(genes[14].min(), genes[14].value(), genes[15].value())
    curt_g["C_MEDIA"] = MembershipFunc(genes[16].value(), genes[17].value(), genes[18].value())
    curt_g["C_ALTA"] = MembershipFunc(genes[19].value(), genes[20].value(), genes[20].max())

    ent_g = FuzzyUniverse("Entropy", genes[21].min(), genes[21].max())
    ent_g["E_BAIXA"] = MembershipFunc(genes[21].min(), genes[21].value(), genes[22].value())
    ent_g["E_MEDIA"] = MembershipFunc(genes[23].value(), genes[24].value(), genes[25].value())
    ent_g["E_ALTA"] = MembershipFunc(genes[26].value(), genes[27].value(), genes[27].max())

    return var_g, skew_g, curt_g, ent_g

def check_gene_values(gene):
    """ Método para verificar se os valores num gene respeitam os intervalos de classe alta, média
    e baixa, com as classes se misturando para gerar as funções Fuzzy.

    Params:
        genes (gene): gene completo que será separado em múltiplos genes.

    Returns:
        Boolean: resultado booleano se a gene respeita ou não os intervalos Fuzzy.
    """
    if (gene[0].value() > gene[1].value()) or (gene[1].value() < gene[2].value()):
        return False
    if (gene[2].value() > gene[3].value() > gene[4].value()) or (gene[4].value() < gene[5].value()):
        return False
    if gene[5].value() > gene[6].value():
        return False
    if gene[0].value() > gene[3].value() > gene[6].value():
        return False
    return True

def fitness_function(chromosome, data):
    """ Método para calcular o fitness de um cromossomo no Algoritmo Genético.

    Params:
        chromosomo (chromosome): Cromossomo que será utilizado para calcular o fitness.
        data (DataFrame): Dados que serão empregados para cáculo do fitness.

    Returns:
        Integer: valor do fitness do cromossomo.
    """
    genes = chromosome.genes

    if len(genes) != 28:
        return 0
    var_check = check_gene_values(genes[:7])
    skew_check = check_gene_values(genes[7:14])
    curt_check = check_gene_values(genes[14:21])
    ent_check = check_gene_values(genes[21:])

    if not (var_check and skew_check and curt_check and ent_check):
        return 0

    var_g_fit, skew_g_fit, curt_g_fit, ent_g_fit = split_genes_to_fuzzy(genes)
    wmcls_fit = WangMendelClassifier(variance=var_g_fit,
                                     skewness=skew_g_fit,
                                     curtosis=curt_g_fit,
                                     entropy=ent_g_fit)
    wmcls_fit.train(data.to_dict('records'), out_label='class', debug=False)
    fit = wmcls_fit.get_fitness() * 100
    return fit

def create_gene(data, variable):
    """ Método criar um gene aleatório de uma variável a partir de um conjunto de dados.

    Params:
        data (DataFrame): Dados que serão empregados para gerar o gene aleatório.
        variable (str): Nome da variável dentro do cojunto de dados.

    Returns:
        Gene: Objeto do tipo Gene.
    """
    data_min = data[variable].min()
    data_max = data[variable].max()
    n_gene_temp = [random.uniform(data_min, data_max) for i in range(0, 7)]
    n_gene_temp.sort()
    gene_temp = [Gene(data_min, data_max, n) for n in n_gene_temp]
    gene_temp[1], gene_temp[2] = gene_temp[2], gene_temp[1]
    gene_temp[4], gene_temp[5] = gene_temp[5], gene_temp[4]
    return gene_temp

def aplicar_fuzzy(nome, data_frame, mamdani):
    """ Método para aplicar as regras fuzzy criadas anteriormente num determinado
    conjunto de dados.

    Params:
        nome (str): Nome do conjunto de dados.
        data_frame (DataFrame): DataFrame contendo os dados a serem aplicados as regras fuzzy.
        mamdani (Mamdani): Classe para a aplicaçãi da inferênia Fuzzy sobre os dados.
    """
    # Aplicação do método e regras para o conjunto de validação.
    records = data_frame.to_dict('records')
    lista = list()
    total = 0
    total_correto = 0
    for rec in records:
        elem = list()
        fz_res = round(mamdani(x1=rec["variance"],
                               x2=rec["skewness"],
                               x3=rec["curtosis"],
                               x4=rec["entropy"]), 0)
        elem.append(rec["variance"])
        elem.append(rec["skewness"])
        elem.append(rec["curtosis"])
        elem.append(rec["entropy"])
        elem.append(rec["class"])
        elem.append(fz_res)
        lista.append(elem)
        if rec["class"] == fz_res:
            total_correto += 1
        total += 1
    # Cálculo e impressão do Percentual de Classe Correta - PCO.
    pco = 100 * (total_correto/total)
    print(str(round(pco, 2)))
    # Geração do arquivo de saída.
    saida_df = pd.DataFrame(lista)
    saida_df.columns = ["variance", "skewness",
                        "curtosis", "entropy", "class", "result"]
    # Registro da hora de início para a geração dos arquivos de saída em pasta específica.
    timestamp = str(datetime.today().strftime('%Y%m%d_%H%M%S'))
    saida_df.to_csv(timestamp + "_" + nome + ".csv", index=False)

def aut_bancaria_fuzzy(arquivo, max_gen, n_pop, p_mut, p_crossover):
    """ Método que detecta autenticidade de notas bancárias.
        Este método usa um alg. genético para definir os melhores intervalos das classes Fuzzy.

    Params:
        arquivo (str): Nome do arquivo com os dados de entrada.
        max_gen (int): Número máximo de gerações para o alg. genético.
        n_pop (int): Tamanho da população para o alg. genético.
        p_mut (float): probabilidade de mutação para o alg. genético.
        p_crossover (float): taxa de crossover para o alg. genético.
    """
    # Leitura do arquivo com os dados de entrada.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df_data = pd.read_csv(dir_path + "/" + arquivo)
    # Separando valores das notas falsas (1) e verdadeiras (0).
    df_f = df_data[df_data["class"] == 1]
    df_v = df_data[df_data["class"] == 0]
    # Criação dos conjuntos de treinamento, validação e de testes.
    # Separação 80% e 20% entre treinamento/validação e testes.
    array_train_val_f, array_test_f = train_test_split(df_f.to_numpy(), test_size=0.2)
    array_train_val_v, array_test_v = train_test_split(df_v.to_numpy(), test_size=0.2)
    # Separação 80% e 20% entre treinamento e validação.
    df_train_val_f = pd.DataFrame(data=array_train_val_f, columns=df_data.columns)
    array_train_f, array_validate_f = train_test_split(df_train_val_f.to_numpy(), test_size=0.2)
    df_train_val_v = pd.DataFrame(data=array_train_val_v, columns=df_data.columns)
    array_train_v, array_validate_v = train_test_split(df_train_val_v.to_numpy(), test_size=0.2)
    # Junção dos dados de V e F para criação dos DataFrames de treinamento, validação e testes.
    array_train = np.concatenate((array_train_f, array_train_v), axis=0)
    array_validate = np.concatenate((array_validate_f, array_validate_v), axis=0)
    array_test = np.concatenate((array_test_f, array_test_v), axis=0)
    df_train = pd.DataFrame(data=array_train, columns=df_data.columns)
    df_validate = pd.DataFrame(data=array_validate, columns=df_data.columns)
    df_test = pd.DataFrame(data=array_test, columns=df_data.columns)

    # Execução de um Algoritmo Genético para definir os limites de separação das classes.
    # Criação dos primeiros genes do Algoritmo Genético.
    variance_gene = create_gene(df_train, "variance")
    skewness_gene = create_gene(df_train, "skewness")
    curtosis_gene = create_gene(df_train, "curtosis")
    entropy_gene = create_gene(df_train, "entropy")
    # Execução do Algoritmo Genético.
    pop = Population(Chromosome(fitness_function,
                                df_train,
                                *(variance_gene + skewness_gene + curtosis_gene + entropy_gene)),
                     n_pop)
    genetic_alg = Algorithm(pop, maxgen=max_gen, mutation=p_mut, crossover=p_crossover)
    genetic_alg.run(debug=True)
    # Seleção do melhor indivíduo.
    best = pop.best()
    # Recuperaçao dos genes com as separações.
    variance_gene, skewness_gene, curtosis_gene, entropy_gene = split_genes_to_fuzzy(
        best.genes)

    # Aplicação do método de Wang Mendel para criar conjunto de regras.
    wmcls = WangMendelClassifier(variance=variance_gene,
                                 skewness=skewness_gene,
                                 curtosis=curtosis_gene,
                                 entropy=entropy_gene)
    rules = wmcls.train(df_train.to_dict('records'),
                        out_label='class', debug=False)

    # Criação da Função Fuzzy para os dados de saída.
    class_fz = FuzzyUniverse("Autenticidade", 0, 1)
    class_fz["0.0"] = MembershipFunc(0, 0, 0.50)
    class_fz["1.0"] = MembershipFunc(0.50, 0.50, 1.00)
    # Instanciação do objeto de inferência Fuzzy.
    mamdani = MamdaniModel(oper=EnumMamdaniOper.Prod,
                           impl=EnumMamdaniImpl.ConjPro,
                           aggr=EnumMamdaniAggr.Max,
                           dfzz=EnumMamdaniDfzz.MoM,
                           x1=variance_gene,
                           x2=skewness_gene,
                           x3=curtosis_gene,
                           x4=entropy_gene,
                           out=class_fz)
    # Inserção das regras definidas anteriormente.
    for rule in rules:
        mamdani.create_rule(x1=rule["antecedent"][0]["label"],
                            x2=rule["antecedent"][1]["label"],
                            x3=rule["antecedent"][2]["label"],
                            x4=rule["antecedent"][3]["label"],
                            out=str(rule["consequent"]))

    # Aplicação das regras Fuzzy selecionada para o conjunto de validação.
    aplicar_fuzzy("validação", df_validate, mamdani)
    # Aplicação das regras Fuzzy selecionada para o conjunto de testes.
    aplicar_fuzzy("teste", df_test, mamdani)

if __name__ == "__main__":
    # Leitura dos dados de entrada.
    ARQUIVO = "entrada.txt"
    # Tamanho máximo da população.
    NP = 50
    # Número máximo de gerações.
    MAX_GEN = 1
    # Probabilidade de mutação.
    PM = 0.05
    # Probabilidade de Crossover/reprodução entre membros de uma geração.
    PC = 0.5
    # Execução do método principal.
    aut_bancaria_fuzzy(ARQUIVO, MAX_GEN, NP, PM, PC)
