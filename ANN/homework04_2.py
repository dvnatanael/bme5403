# Numerical Operations
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import sys
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter

# Timing
import time
from datetime import datetime

# Beep
import winsound

# Custom Utility Functions
import util

start = time.perf_counter()
print(f'start: {datetime.now().strftime("%Y%m%d %H.%M.%S")}')


class CamelbackDataset(Dataset):
    """A six-hump camelback function dataset class.
    x: features.
    y: targets, if None, do prediction.
    """

    def __init__(self, x, y=None):
        self.x = torch.FloatTensor(x)
        self.y = y if y is None else torch.FloatTensor(y)

    def __getitem__(self, i):
        if self.y is None:
            return self.x[i]
        else:
            return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)


class ANN(nn.Module):
    """A neural network model to train the CamelbackDataset on."""

    def __init__(self, input_dim):
        super(ANN, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)  # (B, 1) -> (B)


def select_feat(train_data, valid_data, test_data, select_all=True):
    """Selects useful features to perform regression."""

    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test \
        = train_data[:, :-1], valid_data[:, :-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = range(1, 3)

    return (raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx],
            raw_x_test[:, feat_idx], y_train, y_valid)


def trainer(train_loader, valid_loader, model, config, device):
    """Training loop"""

    # Define your loss function.
    criterion = nn.MSELoss(reduction='mean')

    # Define your optimization algorithm.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    optimizers = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam,
        'Adadelta': torch.optim.Adadelta
    }
    optimizer = 'Adadelta'
    optimizer = optimizers[optimizer](model.parameters(), **config[optimizer])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', **config['lr_plateau'], verbose=True)

    if not os.path.isdir('./homework04_2/runs'):
        os.makedirs('./homework04_2/runs')

    # Writer of tensorboard.
    writer = SummaryWriter(log_dir=f'./homework04_2/runs/{datetime.now().strftime("%Y%m%d%H%M")}')

    if not os.path.isdir('./homework04_2/models'):
        os.makedirs('./homework04_2/models')  # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], np.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()  # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]:',
              f'Train loss: {mean_train_loss:.4f},',
              f'Valid loss: {mean_valid_loss:.4f}', file=sys.stderr)
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if scheduler is not None:
            scheduler.step(mean_valid_loss)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(),
                       config['save_path'])  # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            break
    # print(model.state_dict())


"""# Configurations
config contains hyper-parameters for training and the path to save your model.
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
# TODO config
config = {
    'seed': 191415879,  # Your seed number, you can pick your lucky number. :)
    'select_all': False,  # Whether to use all features.
    'valid_ratio': 2 / 9,  # validation_size = train_size * valid_ratio
    'n_epochs': 5000,  # Number of epochs.
    'batch_size': 512,
    'SGD': {
        'lr': 1e-4,
        'momentum': 0.9
    },
    'Adam': {
        'lr': 1e-3,
        'betas': (0.8, 0.999),
        'eps': 1e-8,
        'weight_decay': 1e-1,
        'amsgrad': False
    },
    'Adadelta': {
        'rho': 0.995,
        'eps': 1e-12,
        'lr': 4,
        'weight_decay': 1e-2
    },
    'lr_plateau': {
        'factor': 0.5,
        'patience': 60,
        # 'cooldown': 30,
        'min_lr': 1e-16
    },
    'early_stop': 400,  # stop if model has not improved for this many consecutive epochs.
    'save_path': './homework04_2/models/model.ckpt'  # Your model will be saved here.
}

"""# Dataloader
Read data from files and set up training, validation, and testing sets.
"""

# Set seed for reproducibility
util.same_seed(config['seed'])

# TODO edit comments
# train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days)
# test_data size: 1078 x 117 (without last day's positive rate)
train_data = pd.read_csv('./camelback.train.csv').values
test_data = pd.read_csv('./camelback.test.csv').values
train_data, valid_data = util.train_valid_split(train_data, config['valid_ratio'], config['seed'])

# Print out the data size.
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

# Select features
x_train, x_valid, x_test, y_train, y_valid \
    = select_feat(train_data, valid_data, test_data, config['select_all'])

# Print out the number of features.
print(f'number of features: {x_train.shape[1]}')

train_dataset, valid_dataset, test_dataset = (
    CamelbackDataset(x_train, y_train),
    CamelbackDataset(x_valid, y_valid),
    CamelbackDataset(x_test)
)

# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                          shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'],
                          shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                         shuffle=False, pin_memory=True)

"""# Start training!"""

# put your model and data on the same computation device.
model = ANN(input_dim=x_train.shape[1]).to(device)
trainer(train_loader, valid_loader, model, config, device)

"""# Testing
The predictions of your model on testing set will be stored at
homework04_2_pred{datetime.now()}.csv.
"""

if not os.path.isdir('./homework04_2/predictions'):
    os.makedirs('./homework04_2/predictions')

model = ANN(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = util.predict(test_loader, model, device)
util.save_pred(
    preds,
    f'./homework04_2/predictions/pred{datetime.now().strftime("%Y%m%d%H%M")}.csv'
)

"""# Reference
This notebook uses code written by Heng-Jui Chang @ NTUEE
(https://github.com/ga642381/ML2021-Spring/blob/main/HW01/HW01.ipynb)
"""

print(f'Time elapsed: {time.perf_counter() - start}')

winsound.Beep(440, 500)
