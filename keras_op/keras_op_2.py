# -*- coding: utf-8 -*-

## Access Google Drive Store

import os
import time
import cv2
import secrets
import shutil
import argparse
import gc
from random import randint
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks

from gapcv.vision import Images

from sklearn.utils import class_weight

print('tensorflow version: ', tf.__version__)

print('keras version: ', tf.keras.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# from matplotlib import pyplot as plt
# %matplotlib inline


## Utils Functions ##

def arg_parameters():
    ## parse command line options
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument("--data", help="Name of data set file without extension")
    parser.add_argument("--evals", default=10, type=int, help="Max number of evaluations")
    args = parser.parse_args()
    return args


def elapsed(start):
    """
    Returns elapsed time in hh:mm:ss format from start time in unix format
    """
    elapsed = time.time()-start
    return time.strftime("%H:%M:%S", time.gmtime(elapsed))


## Parameters GAP ##

def gap():
    args = arg_parameters()
    data_set = args.data
    if not os.path.isfile('{}.h5'.format(data_set)):
        images = Images(data_set, 'plants_photos', config=['resize=(128,128)', 'store', 'stream'])

    # load plants.h5 if exist
    images = Images(config=['stream'], augment=['flip=horizontal', 'edge', 'zoom=0.3', 'denoise'])
    images.load(data_set)
    return images


def gap_generator(minibatch, images):
    images.minibatch = minibatch
    gap_generator = images.minibatch
    return gap_generator


def data():
    images = gap()

    images.split = 0.2
    X_test, Y_test = images.test

    Y_int = [y.argmax() for y in Y_test]
    class_weights = class_weight.compute_class_weight(
        'balanced',
        np.unique(Y_int),
        Y_int
    )

    generator = gap_generator

    return generator, X_test, Y_test, class_weights, images

def space():
    return {
        'filter': np.random.choice([32, 64, 128, 256, 512]),
        'kernel': np.random.choice([3, 4]),
        'activation_0': np.random.choice(['relu', 'tanh']),
        'activation_1': np.random.choice(['softmax','sigmoid']),
        'range': np.random.choice([2, 3]),
        'optimizer': np.random.choice(['adam', 'sgd']),
        'dropout': np.random.uniform(0, 0.5),
        'minibatch': np.random.choice([16, 32, 64, 128]),
        'conditional': np.random.choice([True, False])
    }

class StopTraining(callbacks.Callback):
def __init__(self, monitor='val_loss', patience=10, goal=0.5):
    self.monitor = monitor
    self.patience = patience
    self.goal = goal

def on_epoch_end(self, epoch, logs={}):
    current_val_acc = logs.get(self.monitor)

    if current_val_acc < self.goal and epoch == self.patience:
        self.model.stop_training = True

## Hyperas ##

def model_op(generator, X_test, Y_test, class_weights, images):

    args = arg_parameters()
    data_set = args.data
    model_name = secrets.token_hex(6)
    total_train_images = images.count - len(X_test)
    n_classes = len(images.classes)
    log = {'model_name': model_name}
    space = space()

    try:
        model = Sequential()

        filter_0 = space['filter']
        log['filter_0'] = filter_0
        kernel_0 = space['kernel']
        log['kernel_0'] = kernel_0
        activation_0 = space['activation']
        log['activation_0'] = activation_0
        model.add(layers.Conv2D(
            filters=filter_0,
            kernel_size=kernel_0,
            activation=activation_0,
            input_shape=(128,128,3)
            )
        )

        # conditional_0 = space['conditional']
        # log['conditional_0'] = conditional_0
        # if conditional_0:
        #     layers.BatchNormalization()
        # activity_regularizer=regularizers.l1(0.001)
        # kernel_regularizer=regularizers.l2(0.001)

        conditional_0 = space['conditional']
        log['conditional_0'] = conditional_0
        if conditional_0:
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        conditional_1 = space['conditional']
        log['conditional_1'] = conditional_1
        if conditional_1:
            dropout_0 = space['dropout']
            log['dropout_0'] = dropout_0
            model.add(layers.Dropout(dropout_0))

        range_0 = space['range']
        log['range_0'] = range_0
        for i, _ in enumerate(range(range_0), 1):
            filters = space['filter']
            log["filters_{}".format(i)] = filters
            kernel_sizes = space['kernel']
            log["kernel_sizes_{}".format(i)] = kernel_sizes
            activations = space['activation_0']
            log["activations_{}".format(i)] = activations
            model.add(layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_sizes,
                    activation=activations
                )
            )

            conditionals_0 = space['conditional']
            log["conditionals_0_{}".format(i)] = conditionals_0
            if conditionals_0:
                model.add(layers.MaxPooling2D(pool_size=(2, 2)))

            conditionals_1 = space['conditional']
            log["conditionals_1_{}".format(i)] = conditionals_1
            if conditionals_1:
                dropouts = space['dropout']
                log["dropouts_{}".format(i)] = dropouts
                model.add(layers.Dropout(dropouts))

        model.add(layers.Flatten())

        conditional_2 = space['conditional']
        log['conditional_2'] = conditional_2
        if conditional_2:
            filter_1 = space['filter']
            log['filter_1'] = filter_1
            activation_1 = space['activation_0']
            log['activation_1'] = activation_1
            model.add(layers.Dense(filter_1, activation=activation_1))

        dropout_1 = space['dropout']
        log['dropout_1'] = dropout_1
        model.add(layers.Dropout(dropout_1))

        activation_2 = space['activation_2']
        log['activation_2'] = activation_2
        model.add(layers.Dense(n_classes, activation=activation_2))

        optimizer = space['optimizer']
        log['optimizer'] = optimizer
        model.compile(
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            optimizer=optimizer
        )

        # callbacks

        earlystopping = callbacks.EarlyStopping(monitor='val_loss', patience=5)
        stoptraining = StopTraining(monitor='val_accuracy', patience=30, goal=0.6)
        model_file = '{}/model/{}.h5'.format(data_set, model_name)
        model_checkpoint = callbacks.ModelCheckpoint(
            model_file,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max'
        )
        log_dir="{}/logs/fit/{}".format(data_set, model_name)
        tensorboard = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        lr_reducer_factor = space['dropout']
        log['lr_reducer_factor'] = lr_reducer_factor
        lr_reducer = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=lr_reducer_factor,
            cooldown=0,
            patience=5,
            min_lr=5e-7
        )

        minibatch_size = space['minibatch']
        log['minibatch_size'] = minibatch_size

        history = model.fit_generator(
            generator=generator(minibatch_size, images),
            validation_data=(X_test, Y_test),
            epochs=100,
            steps_per_epoch=int(total_train_images / minibatch_size),
            initial_epoch=0,
            verbose=0,
            class_weight=class_weights,
            # max_queue_size=20,
            # workers=24,
            # use_multiprocessing=True,
            callbacks=[
                model_checkpoint,
                earlystopping,
                tensorboard,
                # lr_reducer,
                stoptraining 
            ]
        )

        model = load_model(model_file)

        score, acc = model.evaluate(X_test, Y_test, verbose=0)
        print('Test accuracy:', acc)
        if acc > 0.85:
            print('log:', log)
            STATUS_OK = True
        else:
            os.remove(model_file)
            shutil.rmtree(log_dir)
            STATUS_OK = False
			
    except Exception as e:
        acc = 0.0
        model = Sequential()
        print('failed', e)
		
    K.clear_session()
    gc.collect()

    return {
        'loss': -acc,
        'status': STATUS_OK,
        'model': model_name
    }


if __name__ == "__main__":

    args = arg_parameters()
    data_set = args.data

    ## Folders Structure ###

    # os.chdir('/content/drive/My Drive')
    # os.makedirs('gap', exist_ok=True)
    # os.chdir('gap')
    os.makedirs('{}/model'.format(data_set), exist_ok=True)

    ## clean tensorboard logs dir ##
    logs_tensorboard = "{}/logs/fit".format(data_set)
    if os.path.isdir(logs_tensorboard):
        for item in os.scandir(logs_tensorboard):
            shutil.rmtree(item.path)

    max_evals=args.evals

    generator, X_test, Y_test, class_weights, images = data()
    
    best_results = []
    for _ in range(max_evals):
        results = model_op(generator, X_test, Y_test, class_weights, images)
        if results['status'] == True:
            best_results.append(results)

    acc_min = 0
    for model_opted in best_results:
        if model_opted['loss'] < acc_min:
            acc_min = model_opted['loss']
            best_model = model_opted['model']

    print("Evalutation of best performing model:", best_model)
    model_file = '{}/model/{}.h5'.format(data_set, best_model)
    best_model = load_model(model_file)
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
