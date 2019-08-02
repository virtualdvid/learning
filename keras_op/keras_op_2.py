# -*- coding: utf-8 -*-

## Access Google Drive Store

import os
import time
import cv2
import secrets
import shutil
import argparse
import gc
from tqdm import tqdm, trange
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

def space_dict():
    return {
        'filter': int(np.random.choice([32, 64, 128, 256, 512])),
        'kernel': int(np.random.choice([3, 4])),
        'activation_0': np.random.choice(['relu', 'tanh']),
        'activation_1': np.random.choice(['softmax','sigmoid']),
        'range': int(np.random.choice([2, 3])),
        'optimizer': np.random.choice(['adam', 'sgd']),
        'dropout': np.random.uniform(0, 0.5),
        'minibatch': int(np.random.choice([16, 32, 64, 128])),
        'conditional': np.random.choice([True, False]),
        'lr_reducer_factor': np.random.uniform(0, 1),
    }

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

    model_name = secrets.token_hex(6)
    log = {'model_name': model_name}
    space = space_dict()

    try:
        model = Sequential()

        log['filter_0'] = space['filter']
        log['kernel_0'] = space['kernel']
        log['activation_0'] = space['activation_0']
        model.add(layers.Conv2D(
            filters=log['filter_0'],
            kernel_size=log['kernel_0'],
            activation=log['activation_0'],
            input_shape=(128,128,3)
            )
        )

        log['conditional_0'] = space['conditional']
        if log['conditional_0']:
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        log['conditional_1'] = space['conditional']
        if log['conditional_1']:
            log['dropout_0'] = space['dropout']
            model.add(layers.Dropout(log['dropout_0']))

        log['range_0'] = space['range']
        for i, _ in enumerate(range(log['range_0']), 1):
            log["filters_{}".format(i)] = space['filter']
            log["kernel_sizes_{}".format(i)] = space['kernel']
            log["activations_{}".format(i)] = space['activation_0']
            model.add(layers.Conv2D(
                    filters=log["filters_{}".format(i)],
                    kernel_size=log["kernel_sizes_{}".format(i)],
                    activation=log["activations_{}".format(i)]
                )
            )

            log["conditionals_0_{}".format(i)] = space['conditional']
            if log["conditionals_0_{}".format(i)]:
                model.add(layers.MaxPooling2D(pool_size=(2, 2)))

            log["conditionals_1_{}".format(i)] = space['conditional']
            if log["conditionals_1_{}".format(i)]:
                log["dropouts_{}".format(i)] = space['dropout']
                model.add(layers.Dropout(log["dropouts_{}".format(i)]))

        model.add(layers.Flatten())

        log['conditional_2'] = space['conditional']
        if log['conditional_2']:
            log['filter_1'] = space['filter']
            log['activation_1'] = space['activation_0']
            model.add(layers.Dense(log['filter_1'], activation=log['activation_1']))

        log['dropout_1'] = space['dropout']
        model.add(layers.Dropout(log['dropout_1']))

        log['activation_2'] = space['activation_1']
        model.add(layers.Dense(n_classes, activation=log['activation_2']))

        log['optimizer'] = space['optimizer']
        model.compile(
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            optimizer=log['optimizer']
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

        log['lr_reducer_factor'] = space['lr_reducer_factor']
        lr_reducer = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=log['lr_reducer_factor'],
            cooldown=0,
            patience=5,
            min_lr=5e-7
        )

        log['minibatch_size'] = space['minibatch']

        model.fit_generator(
            generator=generator(log['minibatch_size'], images),
            validation_data=(X_test, Y_test),
            epochs=100,
            steps_per_epoch=int(total_train_images / log['minibatch_size']),
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
        STATUS_OK = False
        print('failed', e)
		
    del log
    K.clear_session()
    for _ in range(12):
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

    total_train_images = images.count - len(X_test)
    n_classes = len(images.classes)
    
    best_results = []

    with trange(max_evals, postfix='best loss: ?') as pbar:
        for _ in pbar:
            results = model_op(generator, X_test, Y_test, class_weights, images)
            if results['status'] == True:
                best_results.append(results)
                try:
                    best_loss = min([best_result['loss'] for best_result in best_results])
                    pbar.postfix = 'best loss: {} {}'.format(str(best_loss), log)
                except:
                    pass

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

    # conditional_0 = space['conditional']
    # log['conditional_0'] = conditional_0
    # if conditional_0:
    #     layers.BatchNormalization()
    # activity_regularizer=regularizers.l1(0.001)
    # kernel_regularizer=regularizers.l2(0.001)
