import numpy as np
import pandas as pd


def main(x_range=(-5, 5), y_range=(-5, 5), data_points=100):
    def f(x, y):
        return (4 - 2.1 * x ** 2 + (x ** 4) / 3) * x ** 2 \
               + x * y \
               + (-4 + 4 * y ** 2) * y ** 2

    x = np.linspace(*x_range, data_points)
    y = np.linspace(*y_range, data_points)
    x, y = np.meshgrid(x, y)
    z = f(x, y)

    df = pd.DataFrame(
        np.vstack([x.reshape(-1), y.reshape(-1), z.reshape(-1)]).T,
        columns=list('xyz'))
    df = df.sample(frac=1).reset_index(drop=True)

    test_ratio = 0.1
    training_set = df[:int((1 - test_ratio) * len(df))]
    training_set.index.name = 'id'
    testing_set = df[int((1 - test_ratio) * len(df)):]
    testing_set = testing_set.reset_index(drop=True)
    testing_set.index.name = 'id'

    training_set.to_csv('camelback.train.csv')
    testing_set.to_csv('camelback.test.csv')


if __name__ == '__main__':
    main(data_points=250)
