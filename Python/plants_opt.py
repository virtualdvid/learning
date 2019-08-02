# -*- coding: utf-8 -*-

"""## Access Google Drive Store"""

user = 'david'

import os
import time
import cv2
import pprint
from random import randint
import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks

from hyperopt import Trials, STATUS_OK, tpe, hp
from hyperas import optim
from hyperas.distributions import choice, uniform # , conditional

from sklearn.utils import class_weight

print('tensorflow version: ', tf.__version__)
print('keras version: ', tf.keras.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# from matplotlib import pyplot as plt
# # %matplotlib inline

"""## Folders Structure"""

# os.chdir('/content/drive/My Drive')
os.makedirs('gap', exist_ok=True)
os.chdir('gap')
os.makedirs('{}/model'.format(user), exist_ok=True)

"""## Import Gap Vision modules"""

from gapcv.vision import Images

"""## Utils Functions"""

def elapsed(start):
    """
    Returns elapsed time in hh:mm:ss format from start time in unix format
    """
    elapsed = time.time()-start
    return time.strftime("%H:%M:%S", time.gmtime(elapsed))

"""## Create or Load images dataset using Image() module

## Parameters GAP
"""

def gap():
    if not os.path.isfile('plants_128.h5'):
        images = Images('plants_128', 'plants_photos', config=['resize=(128,128)', 'store', 'stream'])

    # load plants.h5 if exist
    images = Images(config=['stream'], augment=['flip=both', 'edge', 'zoom=0.3', 'denoise'])
    images.load('plants_128')
    return images

def gap_generator(minibatch, images):
    images.minibatch = minibatch
    gap_generator = images.minibatch
    return gap_generator

def data(user):
    images = gap()
    
    images.split = 0.2
    X_test, Y_test = images.test
    
    total_train_images = images.count - len(X_test)
    
    Y_int = [y.argmax() for y in Y_test]
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(Y_int),
                                                      Y_int)
    
    generator = gap_generator
    
    return generator, X_test, Y_test, class_weights, images, total_train_images, user

# Clear any logs from previous runs
!rm -R ./$user/logs/ # rf

"""# Hyperas"""

def model_op(gap_generator, X_test, Y_test, class_weights, images, total_train_images, user):
    user = 'david'
    log = {}
    try:
        model = Sequential()
        
        filter_0 = {{choice([32, 64, 128, 256, 512])}}
        log['filter_0'] = filter_0
        kernel_0 = {{choice([3, 4])}}
        log['kernel_0'] = kernel_0
        activation_0 = {{choice(['relu', 'tanh'])}}
        log['activation_0'] = activation_0
        model.add(layers.Conv2D(filters=filter_0, kernel_size=kernel_0, activation=activation_0, input_shape=(128,128,3)))
        
        conditional_0 = {{choice([True, False])}}
        log['conditional_0'] = conditional_0
        if conditional_0:
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            
        conditional_1 = {{choice([True, False])}}
        log['conditional_1'] = conditional_1
        if conditional_1:
            dropout_0 = {{uniform(0, .9)}}
            log['dropout_0'] = dropout_0
            model.add(layers.Dropout(dropout_0))
            
        range_0 = {{choice([0, 2, 3, 4, 5])}}
        log['range_0'] = range_0
        for i, _ in enumerate(range(range_0), 1):
            filters = {{choice([32, 64, 128, 256, 512])}}
            log["filters_{}".format(i)] = filters
            kernel_sizes = {{choice([3, 4])}}
            log["kernel_sizes_{}".format(i)] = kernel_sizes
            activations = {{choice(['relu', 'tanh'])}}
            log["activations_{}".format(i)] = activations
            model.add(layers.Conv2D(filters=filters, kernel_size=kernel_sizes, activation=activations))
            
            conditionals_0 = {{choice([True, False])}}
            log["conditionals_0_{}".format(i)] = conditionals_0
            if conditionals_0:
                model.add(layers.MaxPooling2D(pool_size=(2, 2)))
                
            conditionals_1 = {{choice([True, False])}}
            log["conditionals_1_{}".format(i)] = conditionals_1
            if conditionals_1:
                dropouts = {{uniform(0, .9)}}
                log["dropouts_{}".format(i)] = dropouts
                model.add(layers.Dropout(dropouts))

        model.add(layers.Flatten())
           
        conditional_2 = {{choice([True, False])}}
        log['conditional_2'] = conditional_2
        if conditional_2:
            filter_1 = {{choice([32, 64, 128, 256, 512])}}
            log['filter_1'] = filter_1
            activation_1 = {{choice(['relu', 'tanh'])}}
            log['activation_1'] = activation_1
            model.add(layers.Dense(filter_1, activation=activation_1))
            
        dropout_1 = {{uniform(0, .9)}}
        log['dropout_1'] = dropout_1
        model.add(layers.Dropout(dropout_1))
        
        activation_2 = {{choice(['softmax','sigmoid'])}}
        log['activation_2'] = activation_2
        model.add(layers.Dense(13, activation=activation_2))

        optimizer = {{choice(['adam', 'sgd'])}}
        log['optimizer'] = optimizer
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                      optimizer=optimizer)

        earlystopping = callbacks.EarlyStopping(monitor='val_loss', patience=5)
        model_file = '{}/model/model_plants_128.h5'.format(user)
        model_checkpoint = callbacks.ModelCheckpoint(model_file,
                                                     monitor='val_accuracy',
                                                     save_best_only=True,
                                                     save_weights_only=False, mode='max')
        log_dir="{}/logs/fit/{}".format(user, time.strftime("%Y%m%d-%H%M%S", time.gmtime()))
        tensorboard = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        minibatch_size = {{choice([16, 32, 64, 128])}}
        log['minibatch_size'] = minibatch_size

        history = model.fit_generator(generator=gap_generator(minibatch_size, images),
                                      validation_data=(X_test, Y_test),
                                      epochs=100,
                                      steps_per_epoch=int(total_train_images / minibatch_size),
                                      initial_epoch=0,
                                      verbose=0,
                                      class_weight=class_weights,
                                      callbacks=[model_checkpoint, earlystopping, tensorboard])
        
        model = load_model(model_file)

        score, acc = model.evaluate(X_test, Y_test, verbose=0)
        print('log:', log)
        print('Test accuracy:', acc)
    except Exception as e:
        acc = 0.0
        model = Sequential()
        print('failed')
    
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

best_run, best_model = optim.minimize(model=model_op,
                                      data=data,
                                      functions=[gap, gap_generator],
                                      algo=tpe.suggest,
                                      max_evals=100,
                                      trials=Trials(),
                                      notebook_name='plants_opt')


print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, Y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)
