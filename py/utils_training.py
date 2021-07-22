# -*- coding: utf-8 -*-

import glob 
import os
import csv
import sys
import tensorflow as tf
# tf.autograph.set_verbosity(3, True)
import numpy as np
import pandas as pd
from PIL import Image

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class DataGenerator(tf.keras.utils.Sequence):
    #'Generates data for Keras'
    def __init__(
            self,
            list_IDs,
            labels,
            batch_size=32,
            to_fit=True,
            data_path='.',
            shuffle=True,
        ):
        #'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.index = 0
        self.to_fit = to_fit
        self.data_path = data_path
        self.X, self.y = self.read_all_files(range(len(self.list_IDs)))

    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        #'Generate one batch of data'
        # Generate indexes of the batch
        #indexes = [index]
        # Generate data

        indexes = range(index*self.batch_size,(index+1)*self.batch_size)
        
        X, y = self.__data_generation(indexes)

        if self.to_fit:
            return X, y
        else:
            return X

    def get_data(self):
        return np.asarray(self.X), np.asarray(self.y)

    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.index = 0
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def read_all_files(self, indexes):
        # Creating one hot encoding
        n_values = 10

        self.X = [self.read_file(self.list_IDs.iloc[i][0]) for i in indexes]
        self.y = [np.eye(n_values)[self.labels.iloc[i, :]][0] for i in indexes]
        return self.X, self.y

    def read_file(self, name):
        path = self.data_path + "/data/imgs/" + str(name) + ".png"
        img = np.asarray(Image.open(path))
        return img

    def __data_generation(self, indexes):
        #'Generates data containing batch_size samples'
        # Generate data

        X = np.asarray([self.X[i] for i in indexes])
        y = np.asarray([self.y[i] for i in indexes])

        return X, y

def model_encaps(model):

    INIT_LR = 0.01
    
    # TODO: Check optimization algorithm used in literature
    print("[INFO] compiling model...")
    # opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
    opt = tf.keras.optimizers.Adam(lr=INIT_LR)
    # opt = tf.keras.optimizers.Adam()
    
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

