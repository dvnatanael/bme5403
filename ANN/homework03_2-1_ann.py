import numpy as np
from itertools import product, cycle


def main():
    w = np.array((1, -1, 0), dtype=np.float64)
    x = cycle(product((-1, 1), (-1, 1)))
    t = cycle((0, 1, 1, 0))
    eta = 0.1

    for i, (x, t) in enumerate(zip(x, t)):
        if i % 4 == 0: print()
        if i == 52: break

        x = np.array((*x, -1))
        a = np.dot(w, x)
        y = 1 if a > 0 else 0
        d = eta * (t - y)

        dw = d * x

        print(f'& {w[0]:4.1f} & {w[1]:4.1f} & {w[2]:4.1f} & {x[0]:2d} & {x[1]:2d} & {a:4.1f} & {y:2d} & {t:2d} & {d:4.1f} & {dw[0]:4.1f} & {dw[1]:4.1f} & {dw[2]:4.1f}', end=' \\\\\n')

        w += dw


if __name__ == '__main__':
    main()
