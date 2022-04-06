import os
import numpy as np
import pandas as pd

pred_path = './homework04_2/predictions'


def calculate_loss(file):
    y = pd.read_csv('camelback.test.csv', index_col=['id'], usecols=['id', 'z'])
    y_hat = pd.read_csv(os.path.join(pred_path, file), index_col=['id'], usecols=['id', 'z'])

    error = y - y_hat

    loss = np.mean(error['z'] ** 2)

    # print(*list(map(lambda x: x.head(), [y, y_hat, error, error ** 2])), sep='\n\n', end='\n\n')
    print(f'{file}: {loss=}')


if __name__ == '__main__':
    for file in os.listdir(pred_path):
        calculate_loss(file)
