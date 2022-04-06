import os
import numpy as np
import pandas as pd

np.set_printoptions(threshold=1000000)
pred_path = './homework04_2/predictions'


def calculate_loss(file):
    t = pd.read_csv('camelback.test.csv', index_col=['id'], usecols=['id', 'z']).values.squeeze()
    y = pd.read_csv(
        os.path.join(pred_path, file),
        index_col=['id'],
        usecols=['id', 'z']
    ).values.squeeze()
    e = t - y

    mse_loss = np.mean(e ** 2)
    mape_loss = np.mean(np.abs(e / t))

    # print(*list(map(lambda x: x.head(), [t, y, e, e ** 2])), sep='\n\n', end='\n\n')
    print(f'{file}:\n'
          f'    {mse_loss = }\n'
          f'    {mape_loss = }')


if __name__ == '__main__':
    for file in os.listdir(pred_path):
        calculate_loss(file)
