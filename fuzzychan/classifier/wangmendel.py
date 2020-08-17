class WangMendelClassifier(object):
    from fuzzychan.relation import EnumRelation

    def __init__(self,
                 **kwargs):
        """
        Modelo Wang-Mendel de classificacao Fuzzy
        :param kwargs: passar os inputs (Xi)
        """
        from typing import Dict
        from fuzzychan.base import FuzzyUniverse

        self.__input = {}  # type: Dict[str, FuzzyUniverse]
        self.__rule = {}
        self.__rulebase = None

        self.__status = {}
        self.__reset_status()

        # Processar todos universos adicionados a esse modelo
        for var in kwargs:
            # Apenas universos fuzzy sao aceitos como um input/output
            if not isinstance(kwargs[var], FuzzyUniverse):
                raise TypeError
            # Demais serao apenas um input
            self.__input[var] = kwargs[var]

    def __reset_status(self):
        self.__status["accuracy"] = 0
        self.__status["p_accuracy"] = 0
        self.__status["train_size"] = 0

    def print_status(self):
        print('=' * 80)
        print("WangMendel Classifier")
        print("\tTrain samples: %d" % (self.__status['train_size']))
        print("\tRule Base [size=%d]:" % (len(self.__rulebase)))
        print('-' * 80)
        print('\n'.join(str(rule['txt']) for rule in self.__rulebase))
        print('-' * 80)
        print("Accuracy: %.2f%c" % (float(self.__status['accuracy'] * 100), '%'))
        print("Pert Accuracy: %.2f%c" % (float(self.__status['p_accuracy'] * 100), '%'))
        print('=' * 80)

    def get_fitness(self):
        return float(self.__status["accuracy"])

    def __call__(self, *args, **kwargs):
        result = []
        for rule in self.__rulebase:
            result_v = rule['antecedent'](**kwargs)
            result_l = rule['consequent']

            if len(args) > 0 and "debug" in args:
                print('-' * 80)
                print(rule["txt"] + " => " + result_l + "(" + str(result_v) + ")")
            result.append({
                "value": result_v,
                "label": result_l
            })
        if len(args) > 0 and "complete" in args:
            return result
        return max(result, key=lambda x: x["value"])

    def train(self, data, out_label='cls', oper=EnumRelation.Min, debug=False):
        """
        Metodo para input de dados de treino para calibrar o classificador
        :param data: Dados para treinamento
        :param out_label: Label de acesso a classe da amostra
        :param oper: Norma-t para agregar o antecedente
        :param debug: Ativar ou nao os prints do processo de geracao das regras
        :return:
        """
        import copy as cp
        from fuzzychan.relation import FuzzyRelation

        # Criar base de regras
        rules = []
        self.__reset_status()

        """ Processar todos dados passados (dataset) """
        for row in data:
            # Criar uma regra para cada elemento do dataset
            antc = [elem for elem in self.__elem_to_rule(row, out_label)]

            # Criar um texto descritivo da regra para visualizacao posterior
            txt = ' E '.join((elem['name'] + ' eh ' + elem['label']) for elem in antc) + ' ENTAO ' + str(row[out_label])
            txt += '(' + str(WangMendelClassifier.__operation(oper, antc)) + ')'

            # Finalizar criacao da regra desse elemento
            rules.append({
                'antecedent': antc,
                'consequent': row[out_label],
                'txt': txt
            })

        """ Debug apenas, visualizar todas regras geradas (uma para cada elemento) """
        if debug:
            print('-' * 80)
            print('\n'.join(str(rule['txt']) for rule in rules))

        """ Reduzir a base de regras, removendo redundancias e inconsistencias """

        # Criar uma copia da base de regras para ser a versao definitiva
        # Incialmente sera igual ao original, porem serao removidas as regras superfluas
        c_rules = cp.deepcopy(rules)

        # Contagem para status
        num_rules = len(rules)


        # Percorrer a base de regra antiga
        for i in range(len(rules)):
            # Se a regra nao estive mais no banco definitivo, ignorar (nao vai mais entrar)
            if rules[i] not in c_rules:
                continue

            # Usar a regra atual de pivo, e comparar ela com todas apos ela (anteriores ja comparadas)
            # Isso visa otimizar o numero de comparacoes
            for j in range(i + 1, len(rules)):
                # Se a regra a ser comparada nao estive mais no banco definitivo, ignorar (ha uma equivalente a ela)
                if rules[j] not in c_rules:
                    continue

                # Verificar a igualdade do antecedente das duas regras
                if min([(1 if rules[i]['antecedent'][k]['label'] == rules[j]['antecedent'][k]['label'] else 0) for k in
                        range(len(rules[i]['antecedent']))]) == 1:

                    # Para redundancias -> elimitar a de menor importancia, pois a outra representa ela ja
                    # Para inconsistencias -> elimitar a de menor importancia, pois a outra eh mais adequada
                    # Em resumo, nos dois casos remove-se a regra de menor grau de ativacao
                    """ BEGIN: REMOCAO DE ITENS """

                    # Calcular o grau de ativacao de cada regra para decidir qual eh mais relevante
                    mship_i = WangMendelClassifier.__operation(oper, rules[i]['antecedent'])
                    mship_j = WangMendelClassifier.__operation(oper, rules[j]['antecedent'])

                    # Decidir quem remover ...
                    if mship_i > mship_j:
                        # Pivo tem grau maior, assim a comparada eh removida do banco de regras definitivo
                        c_rules.remove(rules[j])
                        # Continuar o processo com o pivo
                    else:
                        # Comparada tem grau maior, assim o pivo eh removido do banco de regras definitivo
                        c_rules.remove(rules[i])
                        # O pivo foi removido, entao devemos passar para o proximo elemento
                        break
                    """ END: REMOCAO DE ITENS """

        """ Construir estrutura de inteferencia """
        rules = []
        for prerule in c_rules:
            params = {}
            for p_antc in prerule["antecedent"]:
                params[p_antc["name"]] = self.__input[p_antc["name"]][p_antc["label"]]
            antecedent = FuzzyRelation(kind=oper, **params)
            rules.append({
                'antecedent': antecedent,
                'consequent': prerule["consequent"],
                'txt': prerule['txt']
            })
        # Atualizar a base de regras desse classificador
        self.__rulebase = rules

        """ Calcular a acuracia do modelo """
        # Contabilizar os erros
        num_miss = 0
        num_pert = 0
        for row in data:
            # Classificar
            r_all = self("complete", **row)
            r = max(r_all, key=lambda x: x["value"])
            # Label diferente?
            if r["label"] != row[out_label]:
                # Contar erro
                num_miss += 1
                num_pert += 1 - max([(ri["value"] if ri["label"] == row[out_label] else 0) for ri in r_all])

        self.__status["accuracy"] = float(num_rules - num_miss) / num_rules
        self.__status["p_accuracy"] = float(num_rules - num_pert) / num_rules
        self.__status["train_size"] = num_rules

        """ Debug apenas, visualizar as regras definitivas (apos a reducao) """
        if debug:
            self.print_status()

        return rules

    def __elem_to_rule(self, elem, out_label):
        """
        Metodo auxiliar para gerar uma estrutura que representa uma regra a partir de um elemento
        :param elem: Coordenadas do dado nos universos desse classificador
        :param out_label: Label de acesso a classe do dado
        :return:
        """
        from fuzzychan.base import FuzzyUniverse, FuzzySet

        # Acessar todos label de universo
        for key_universe in self.__input.keys():

            # Pegar o universo atual
            universe = self.__input[key_universe]  # type: FuzzyUniverse

            # Criar um pedaco antecedente para o universo atual
            # Esses pedacos de antecedente irao compor o antecedente final da regra que representa o dado atual
            p_antc = {}
            p_antc['name'] = key_universe
            p_antc['value'] = elem[key_universe]

            # Acessar todos conjuntos do universo
            for key_set in universe:

                # Pegar o conjunto fuzzy atual
                fuzzy_set = universe[key_set]  # type: FuzzySet
                # Calcular a pertinencia do dado nesse conjunto
                pert_val = fuzzy_set.func(x=elem[key_universe])

                # Pegar o conjunto que de o maior grau de ativacao
                if 'label' not in p_antc.keys() or p_antc['mship'] < pert_val:
                    p_antc['label'] = key_set
                    p_antc['mship'] = pert_val

            # Retornar a estrutura do conjunto de maior grau de ativacao
            yield p_antc

    @staticmethod
    def __operation(oper, rule):
        """
        Metodo para operar o grau de ativacao final de uma regra
        :param oper:
        :param rule:
        :return:
        """
        import numpy as np
        from fuzzychan.relation import EnumRelation

        elems = [elem['mship'] for elem in rule]

        if oper == EnumRelation.Min:
            return min(elems)
        elif oper == EnumRelation.Prod:
            return np.prod(elems)
        return 0
