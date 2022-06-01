# import libraries
import re
import os
import sys
import time
import pickle
from dataclasses import dataclass, field
from json import dumps
from collections import namedtuple
from typing import Generator, Union
from random import randint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend, Input

from tensorflow import test
from keras.callbacks import Callback, History, EarlyStopping, ReduceLROnPlateau
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Model
import keras.optimizers
from keras.optimizers import Optimizer

start = time.perf_counter()
Data = namedtuple('Data', ['x', 'y'])


def setup_logging() -> "logging.Logger":
    """Set up the logger."""
    import logging
    formatter = logging.Formatter('{asctime}|[{levelname:^8s}]|{name}|{message}', style='{')
    formatter.default_time_format = '%Y%m%d %H.%M.%S'
    formatter.default_msec_format = '%s.%03d'

    filename_prefix = (sys.argv[1] + '_') if len(sys.argv) > 1 else ''
    debug_handler = logging.FileHandler(f'{filename_prefix}debug.log')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)

    handler = logging.FileHandler(f'{filename_prefix}info.log')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(debug_handler)
    logger.addHandler(handler)
    return logger


def initialize() -> None:
    """Initialize runtime."""
    import tracemalloc

    logger.info('=' * 78)
    logger.info('{:=^78s}'.format(f'Running {filename}'))
    logger.info('=' * 78)
    logger.debug('Starting tracemalloc.')
    tracemalloc.start()
    np.random.seed(10)

    # setup ./model subdirectory to save and load model if it does not already exist
    subdir = './model'
    try:
        os.mkdir(subdir)
        logger.info(f'Created {subdir} subdirectory.')
    except FileExistsError:
        logger.info(f'Using existing {subdir} subdirectory...')
    except FileNotFoundError:
        assert not os.path.exists(subdir)
        logger.exception(f'Failed to create {subdir} subdirectory...Quitting.')
        cleanup(-1)


def cleanup(code: int, show_figures: bool = False) -> None:
    """Clean up runtime."""
    import tracemalloc
    if tracemalloc.is_tracing():
        curr_memory, peak_memory = tracemalloc.get_traced_memory()
        logger.debug(f'{curr_memory = } bytes')
        logger.debug(f'{peak_memory = } bytes')
        logger.debug('Stopping tracemalloc.')
        tracemalloc.stop()
    logger.info('=' * 78)
    logger.info('{:=^78s}'.format(f'Finished running {filename}. Exiting...'))
    logger.info('=' * 78)
    logger.info(f'[Finished in {time.perf_counter() - start:.1f}s]')

    if show_figures:
        plt.show(block=False)
        plt.pause(5)
        plt.close()

    import winsound
    winsound.Beep(500, 500)
    winsound.Beep(500, 500)
    exit(code)


def preprocess_data(data: Data) -> Data:
    """Preprocess the data.

    Convert image data range from [0, 255] to [0, 1] inclusive.

    One-hot encode labels.
    """
    x_normalized = data.x.astype('float32') / 255
    y_one_hot = np_utils.to_categorical(data.y)
    return Data(x_normalized, y_one_hot)


def get_model_summary(model: Model, line_length: int = 80) -> str:
    """Get the model summary as a string."""
    string_list = []
    model.summary(line_length=line_length, print_fn=lambda s: string_list.append(s))
    return '\n'.join(string_list)


def show_train_history(train_history: dict, *, history_only=False) -> None:
    """Show the model training history."""

    def plotter(train: str, val: str):
        ax.plot(train_history[train], label=f'train')
        ax.plot(train_history[val], label=f'validation')
        ax.set_title('Train History')
        ax.set_ylabel(train.title())
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper left')

    fig: plt.Figure = plt.gcf()
    gs: plt.GridSpec = fig.add_gridspec(4, 6, hspace=0.5, wspace=0.5)
    # plot accuracy
    ax: plt.Axes = fig.add_subplot(gs[:2, :3])
    plotter('accuracy', 'val_accuracy')
    # plot loss
    ax: plt.Axes = fig.add_subplot(gs[:2, 3:])
    plotter('loss', 'val_loss')


def show_prediction(data, predictions, label_dict: dict, index: int = 0,
                    *, prob_dist=None, gs_start_index: int = 0) -> Generator:
    """Show the model predictions."""
    rows, cols = 4, 6
    fig: plt.Figure = plt.gcf()
    gs: plt.GridSpec = fig.add_gridspec(rows, cols, hspace=0.5, wspace=0.5)
    counter = 0
    while True:
        img, label, pred = map(lambda x: x[index], (*data, predictions,))

        if prob_dist is not None:
            prob = prob_dist[index]
            log: dict = {v: float(p) for v, p in zip(label_dict.values(), prob)}
            logger.info(f'label:{label_dict[label[0]]} predict:{label_dict[pred]}')
            logger.info(f'Prediction probabilities:\n{dumps(log, indent=4)}')

        # show the image
        i: int = gs_start_index + counter
        ax: plt.Axes = fig.add_subplot(gs[i // cols, i % cols])
        ax.imshow(img, cmap='binary')  # bitmap image

        title: str = f'{index}, {label_dict[label[0]]}' \
                     + (f'=>{label_dict[pred]}' if len(predictions) else '')

        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        yield counter, index
        counter += 1
        index += 1


def create_model(batch_size: int, input_shape: tuple, num_classes: int, *, loss: str,
                 optimizer: Union[str, Optimizer], load_existing: bool = False) -> "Sequential":
    """Create a Sequential CNN + classifier model."""
    from keras.models import Sequential
    from keras.layers import \
        Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, \
        RandomContrast, RandomFlip, RandomRotation, ZeroPadding2D
    from keras.regularizers import l2

    # Define the model
    # preprocessing layer
    preprocessor = Sequential([
        RandomFlip('horizontal'),
        RandomContrast(0.2),
        RandomRotation(0.02),
    ], name='preprocessing_layer')
    convolution_layer = Sequential([
        # convolution layers
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        # MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
    ], name='convolution_layer')
    classifier = Sequential([
        # classification layers
        Flatten(),  # output shape: (None, 4, 4, 16)
        Dropout(rate=0.50),
        Dense(512, activation='elu', kernel_regularizer=l2(1e-3)),  # hidden layer with 512
        # neurons
        Dropout(rate=0.25),
        Dense(64, activation='elu', kernel_regularizer=l2(1e-3)),
        Dropout(rate=0.25),
        Dense(32, activation='elu', kernel_regularizer=l2(1e-3)),
        # output layer, use softmax to convert network output to probabilities
        Dense(num_classes, activation='softmax'),
    ], name='classification_layer')
    model = Sequential([
        preprocessor,
        convolution_layer,
        classifier
    ], name='cifar_cnn')
    model.build(input_shape=(batch_size, *input_shape))
    # log the model summary
    logger.info(get_model_summary(preprocessor, line_length=79))
    logger.info(get_model_summary(convolution_layer, line_length=79))
    logger.info(get_model_summary(classifier, line_length=79))
    logger.info(get_model_summary(model, line_length=79))

    # load previously trained model
    if load_existing:
        try:
            model.load_weights('./model/cifar10.h5')
            logger.info('Using previously trained model.')
        except FileNotFoundError:
            logger.warning('Model not found or does not exist. Start training a new model.')

    # set the loss function and optimizer
    logger.info(f'Using optimizer {optimizer}')
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


@dataclass
class Record:
    record: list[list[str]] = field(default_factory=list)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.record.clear()
        for _ in range(2):
            self.record.append([])


epoch_record: Record = Record()


class CustomCallback(Callback):
    start: float
    epoch: int = 0

    def __init__(self):
        super().__init__()

    def on_train_begin(self, logs=None):
        logger.info('Training started.')
        self.start = time.perf_counter()
        super().on_train_begin(logs)

    def on_epoch_begin(self, epoch, logs=None):
        global epoch_record
        epoch_record.record[0].append(f'Epoch: {epoch}/{self.params["epochs"]}')
        super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        global epoch_record
        epoch_record.record[0].append(f'{int(time.perf_counter() - self.start):2d}s')
        epoch_record.record[0].extend([f'{k}: {v:.6f}' for k, v in logs.items()])
        self.epoch += 1
        self.start = time.perf_counter()
        super().on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        if self.epoch == self.params['epochs']:
            logger.info(f'Training finished. '
                        f'Model was trained for {self.params["epochs"]} epochs.')


class CustomReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        global epoch_record
        old_lr = backend.get_value(self.model.optimizer.lr)
        str_lr = f'{old_lr:.6f}' if old_lr >= 1e-6 else old_lr
        epoch_record.record[0].append(f'lr: {str_lr}')
        super().on_epoch_end(epoch, logs)
        new_lr = backend.get_value(self.model.optimizer.lr)
        if new_lr != old_lr:
            str_lr = f'{new_lr:.6f}' if new_lr >= 1e-6 else new_lr
            epoch_record.record[1].append(f'Model has not improved for {self.patience} epochs. '
                                          f'Reducing learning rate to {str_lr}.')


class RecordWriter(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        global epoch_record
        logger.debug(' - '.join(epoch_record.record[0]))
        if len(epoch_record.record[1]):
            logger.debug(':'.join(epoch_record.record[1]))
        epoch_record.reset()


class CustomEarlyStopping(EarlyStopping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_end(self, logs=None):
        if self.stopped_epoch:
            logger.info(f'Model has not improved for {self.patience} epochs. Stopping training.')
            logger.info(f'Training finished. Model was trained for {self.stopped_epoch} epochs.')


def train(model: Model, data: Data, config: dict) -> History:
    """Train the model.

    :returns: The train `History` object.
    """
    early_stop = CustomEarlyStopping(**config['early stop'])
    reduce_lr_on_plateau = CustomReduceLROnPlateau(**config['reduce_lr_on_plateau'])
    train_history: History = model.fit(
        data.x, data.y, **config['fit'],
        callbacks=[CustomCallback(), reduce_lr_on_plateau, RecordWriter(), early_stop]
    )
    return train_history


def save_model(dir: str, model: Model, history: History, figure: plt.Figure):
    # save model as json
    with open(os.path.join(dir, 'cifar10.json'), 'w') as f:
        f.write(model.to_json())
        logger.info('Model structure saved successfully.')

    # save weights as h5
    model.save_weights(os.path.join(dir, 'cifar10.h5'))
    logger.info('Model weights saved successfully.')

    # save train history as pkl
    with open(os.path.join(dir, 'cifar10_train_history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
        logger.info('Train history saved successfully.')

    # save figure as png
    figure.savefig(os.path.join(dir, 'cifar10.png'), bbox_inches='tight', transparent=False)


def main(config: dict):
    # load the dataset
    # x.shape: (number of images, image size, number of channels) e.g. (50000, 32, 32, 3)
    # y.shape: (number of images, label) e.g. (50000, 1)
    train_data_raw, test_data_raw = map(lambda data: Data(*data), cifar10.load_data())
    logger.info(f'raw train data:\timages:{train_data_raw.x.shape} '
                f'labels:{train_data_raw.y.shape}')
    logger.info(f' raw test data:\timages:{test_data_raw.x.shape} '
                f'labels:{test_data_raw.y.shape}')

    # preprocess the data
    train_data = preprocess_data(train_data_raw)
    test_data = preprocess_data(test_data_raw)
    logger.debug(f'one-hot encoded labels: {test_data.y.shape}')

    # create the model
    model = create_model(config['fit']['batch_size'], train_data.x.shape[-3:],
                         train_data.y.shape[-1], optimizer=config['optimizer']['adadelta'],
                         **config['model'])
    train_history: History = train(model, train_data, config)  # train the model
    try:
        history: dict = train_history.history
        for k, v in history.items():
            history[k] = list(map(float, v))
        logger.debug(f'train history:\n{dumps(history, indent=4)}')
    except TypeError as e:
        logger.exception(f'train history:\n{train_history.history}')

    fig: plt.Figure = plt.figure(figsize=(12, 8))
    show_train_history(train_history.history)  # show training and validation accuracy and loss

    # Predict test data labels
    predictions: pd.Series = pd.Series(np.argmax(model.predict(test_data.x), axis=-1))
    with pd.option_context('display.max_rows', 10):
        logger.debug(f'predictions:\n{predictions}')

    # Evaluate the model accuracy
    scores = model.evaluate(test_data.x, test_data.y, verbose=1)
    logger.info(f'model accuracy: {scores[1]}')

    # Show prediction results
    label_dict = {
        0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
        5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
    }
    index = randint(0, test_data.x.shape[0] - 10)
    g = show_prediction(test_data_raw, predictions, label_dict, index, gs_start_index=12)
    for _ in zip(range(10), g):
        pass

    # Show prediction probabilities
    pred_probabilities = model.predict(test_data.x)
    index = randint(0, test_data.x.shape[0] - 12)
    next(show_prediction(test_data_raw, predictions, label_dict, 0,
                         prob_dist=pred_probabilities, gs_start_index=22))
    next(show_prediction(test_data_raw, predictions, label_dict, 3,
                         prob_dist=pred_probabilities, gs_start_index=23))

    # Confusion matrix
    logger.debug(f'{predictions.shape = }')
    # to create confusion matrix, data must be a 1d array
    logger.debug(f'{test_data_raw.y.shape = }')
    logger.debug(f'{test_data_raw.y.flatten().shape = }')

    logger.info(f'labels dictionary:\n{dumps(label_dict, indent=4)}')
    confusion_matrix = pd.crosstab(test_data_raw.y.flatten(), predictions, rownames=["label"],
                                   colnames=["predict"])
    logger.info(f'Confusion matrix:\n{confusion_matrix}')

    return model, train_history, fig


logger = setup_logging()
filename = (re.split(r'[\\/]', __file__)[-1])

if __name__ == '__main__':
    initialize()
    logger.info(f'Using GPU {gpu}' if (gpu := test.gpu_device_name()) else f'Using CPU')
    config = {
        'model': {
            'loss': 'categorical_crossentropy',
            'load_existing': False
        },
        'optimizer': {
            'adam': {},
            'adadelta': {
                'learning_rate': 1.,
                'rho': 0.995,
                'epsilon': 1e-7
            },
        },
        'fit': {
            'validation_split': 0.2,
            'epochs': 200,
            'batch_size': 128,
            'verbose': 0
        },
        'early stop': {
            'monitor': 'val_loss',
            'min_delta': 1e-8,
            'patience': 30,
            'verbose': 1,
            'mode': 'auto',
            'baseline': None,
            'restore_best_weights': False
        },
        'reduce_lr_on_plateau': {
            'monitor': 'val_loss',
            'factor': 0.7,
            'min_lr': 1e-8,
            'patience': 5,
            'verbose': 1,
            'cooldown': 10,
        }
    }
    logger.info(f'model config:\n{dumps(config, indent=4)}')
    optimizers = {
        'adam': keras.optimizers.adam_v2.Adam,
        'adadelta': keras.optimizers.adadelta_v2.Adadelta
    }
    for optim, kwargs in config['optimizer'].items():
        config['optimizer'][optim] = optimizers[optim](**kwargs)
    model, train_history, fig = main(config)
    save_model('./model', model, train_history, fig)  # save the model
    cleanup(0, show_figures=True)
