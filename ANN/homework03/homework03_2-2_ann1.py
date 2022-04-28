import sys
import numpy as np
from itertools import product, cycle


class NeuralNetwork:
    ε = 1e-8


    @staticmethod
    def σ(x, λ=1):
        return 1 / (1 + np.exp(-λ * x))


    def __init__(self):
        pass


    def __repr__(self):
        z = [z.T for z in self.z]
        a = [a.T for a in self.a]
        # out = f'{self.depth = }\n{self.structure = }\n{self.w = }\nself.{z = }\nself.{a = }\n{self.δ = }'
        out = f'{self.w = }\nself.{z = }\nself.{a = }\n{self.t = }\n{self.δ = }'
        return out


    def initialize(self):
        self.structure = [2, 2, 1]
        self.depth = len(self.structure)
        self.η = 0.1
        self.w = []
        self.w.append(np.array([[0.2, -0.4, 0.8], [0.2, -0.2, -0.1]], dtype=np.float64))
        self.w.append(np.array([0.1, -0.4, 0.3], dtype=np.float64))
        self.z = []
        self.a = []
        self.δ = []
        self.t = None


    def forward(self, x, file):
        self.z = []
        self.a = []

        a = np.vstack((x, np.asarray(-1)))
        self.z.append(a)
        self.a.append(a)
        for i, wl in enumerate(self.w):
            z = np.matmul(wl, a)
            a = NeuralNetwork.σ(z)

            out = np.ndarray.flatten(wl).tolist()
            if type(t := np.ndarray.flatten(z).tolist()) == list: out.extend(t)
            else: out.append(z)
            if type(t := np.ndarray.flatten(a).tolist()) == list: out.extend(t)
            else: out.append(a)

            out = ' '.join([f'{i:8.5f}' for i in out])
            file.write(out)

            if i != self.depth - 2:
                a = np.vstack((a, np.asarray(-1)))
            self.z.append(z)
            self.a.append(a)


    def calculate_delta(self, t):
        self.t = t
        self.δ = [-self.η * -(t - self.a[-1])]

        δ = np.zeros(1)
        δ[0] += self.δ[-1] * self.a[2][0] * (1 - self.a[2][0])
        self.δ.append(np.expand_dims(δ, axis=0).T)

        δ = np.zeros(2)
        δ[0] += self.δ[-1] * self.a[1][0] * (1 - self.a[1][0]) * self.w[1][0]
        δ[1] += self.δ[-1] * self.a[1][1] * (1 - self.a[1][1]) * self.w[1][1]
        self.δ.append(np.expand_dims(δ, axis=0).T)

        print(self.δ)
        self.δ.reverse()

        # for a, w, n in zip(reversed(self.a), reversed(self.w + [np.asarray([1])]), reversed(self.structure[1:])):
            # prev_δ = []
            # δ = self.δ[-1].copy()
            # a = a.flatten()
            # if w.ndim == 1: w = np.expand_dims(w, axis=0)
            #
            # for wi in w:
            #    print(a, wi)
            #
            # print('', n, a, w, sep='\n\t')
            # print(a.shape, w.shape)


    def update_weights(self, file):
        out = []
        for w, δ, a in zip(self.w, self.δ[:-1], self.a[:-1]):
            δw = (δ * a.T).squeeze()
            out.append(δw.flatten()) 
            w += δw
        file.write(' '.join([f'{i:8.5f}' for i in np.concatenate(out)]))


    def backward(self, t, file):
        self.calculate_delta(t)
        self.update_weights(file)


def main():
    nn = NeuralNetwork()
    nn.initialize()

    x = cycle(product((-1, 1), (-1, 1)))
    t = cycle((0, 1, 1, 0))

    with open('homework03_2-2out1.txt', 'w', encoding='utf8') as f:
        f.write(str(nn))
        f.write('\n\n')

        for i, (x, t) in enumerate(zip(x, t)):
            if i == int(sys.argv[1]): break

            x = np.asarray(x).reshape(2, -1)
            nn.forward(x, file=f)
            f.write('\n')
            nn.backward(t, f)
            f.write('\n')
            f.write(str(nn))
            f.write('\n')
            f.write('\n')

            # print(f'{t=} a={nn.a[-1][0]} e={t - nn.a[-1][0]}')
            # nn.forward(x, file=sys.stdout)
            # print()


if __name__ == '__main__':
    np.set_printoptions(precision=5, suppress=True)
    main()
