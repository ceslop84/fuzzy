# ==================================================================================================================== #
#   Author: Rafael Hideo Toyomoto                                                                                      #
#   Created: 15/09/2018                                                                                                #
# ==================================================================================================================== #

# ==================================================================================================================== #
#   MembershipFunc
# -------------------------------------------------------------------------------------------------------------------- #
#   Classe que representa funcoes de pertinencia (fuzzy)
# ==================================================================================================================== #

from matplotlib.pyplot import Figure


class MembershipFunc(object):

    def __init__(self, *args, **kwargs):
        """
        Funcoes de pertinencia
        :param args:
        :param kwargs:
        """

        # Get 'kind' (default: triangular)
        kind = ''
        if 'kind' not in kwargs.keys():
            if len(args) == 2:
                if isinstance(args[0], float):
                    self._kind = kind = 'rectangular'
                else:
                    self._kind = kind = 'custom'
            elif len(args) == 3:
                self._kind = kind = 'triangular'
            elif len(args) == 4:
                self._kind = kind = 'trapezoidal'
        else:
            self._kind = kind = str(kwargs['kind'])

        self._height = float(kwargs['height']) if 'height' in kwargs.keys() else 1.0
        if kind == 'trapezoidal':
            self._a = float(args[0])
            self._m = float(args[1])
            self._n = float(args[2])
            self._b = float(args[3])
        elif kind == 'triangular':
            self._a = float(args[0])
            self._m = self._n = float(args[1])
            self._b = float(args[2])
        elif kind == 'rectangular' or kind == 'crisp':
            self._a = self._m = float(args[0])
            self._b = self._n = float(args[1])
        elif kind == 'custom':
            self._func = args[0]
            self._ifunc = args[1]
        else:
            raise TypeError("MembershipFunc: invalid 'kind' (trapezoidal/triangular/rectangular|crisp)")

    def __call__(self, *args, **kwargs):
        # Get 'kind'
        kind = self._kind
        # Get 'height'
        height = self._height

        if 'inv' in kwargs.keys() and kwargs['inv'] == 'true':
            # INVERSE OPERATION
            return self.inv(float(kwargs['mi']))

        return self.calc(float(kwargs['x']))

    def calc(self, x):
        # Get 'kind'
        kind = self._kind
        # Get 'height'
        height = self._height

        # Get 'x' value
        if kind == 'trapezoidal' or kind == 'triangular' or kind == 'rectangular' or kind == 'crisp':
            a = self._a
            m = self._m
            n = self._n
            b = self._b
            if a < x < m:
                return height * (x - a) / (m - a)
            elif m <= x <= n:
                return height
            elif n < x < b:
                return height * (b - x) / (b - n)
            return 0

        elif kind == 'custom':
            return self._func(x)

        return 0

    def inv(self, mi):
        # Get 'kind'
        kind = self._kind
        # Get 'height'
        height = self._height

        if kind == 'trapezoidal' or kind == 'triangular' or kind == 'rectangular' or kind == 'crisp':
            a = self._a
            m = self._m
            n = self._n
            b = self._b
            if mi > 1 or mi < 0:
                return [None, None]
            if mi == 0:
                return a, b
            return [float(mi * float(m - a) / height + a), float(b - mi * float(b - n) / height)]

        elif kind == 'custom':
            return self._ifunc(mi)

        return 0

    def alpha_cut(self, alpha, height=1):
        # FIXME: valido apenas para alguns tipos

        # Calculo dos pontos inversos
        p1, p2 = self(mi=alpha, inv='true')

        # Criar uma funcao crisp para representar o alfa corte
        return MembershipFunc(p1, p2, kind='crisp', height=height)

    def support(self):
        # FIXME: valido apenas para alguns tipos
        return MembershipFunc(self._a, self._b, kind='crisp')

    def core(self):
        # FIXME: valido apenas para alguns tipos
        return MembershipFunc(self._m, self._n, kind='crisp')

    def inclusion(self, target, interval):
        """
        Grau de inclusao  entre essa funcao com outra passada
        :param target: Funcao de pertinencia a ser comparado
        :type target: MembershipFunc
        :param interval: Valores de 'x' do universo
        :return:
        """
        card, outer = 0, 0
        for xi in interval:
            # Calcular pertinencia para cada xi
            yi = self(x=xi)
            # Cardinalidade dessa funcao
            card += yi
            # Soma de quanto a funcao 'target' nao preenche essa funcao
            outer += max(0, (yi - target(x=xi)))
        if card == 0:
            return 0
        # Grau de inclusao (normalizada)
        return (card - outer) / card


# ==================================================================================================================== #
