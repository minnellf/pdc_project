# -*- coding: utf-8 -*-

import glob 
import os
import csv
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
# tf.autograph.set_verbosity(3, True)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import utils_training
from datetime import datetime
import model_libs
import random
import threading
import time

physical_devices = tf.config.list_physical_devices()
#physical_devices = tf.config.list_physical_devices('GPU')
print('physical_devices:', physical_devices)

try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  print('Invalid device or cannot modify virtual devices once initialized.')

# datetime object containing current date and time
now = datetime.now()

NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))

if (os.getenv('TRAIN_CLUSTER') != None):
  data_path = os.getenv('TMPDIR')
else:
  data_path = '.' 

train_annot_df = pd.read_csv(data_path + '/data/full.csv', low_memory=False)

source_coordinates = ['id']
target_coordinates = ['class']

dev_train_class, test_class = train_test_split(train_annot_df, test_size=0.1, random_state=1, stratify=train_annot_df['class'])
train_class, val_class = train_test_split(dev_train_class, test_size=(1.0/9.0), random_state=1, stratify=dev_train_class['class'])

X_train = train_class[source_coordinates]
y_train = train_class[target_coordinates]

X_val = val_class[source_coordinates]
y_val = val_class[target_coordinates]

ELAB_IMG_SIZE_X = 32
ELAB_IMG_SIZE_Y = 32

LOAD_MODEL = int(os.getenv('LOAD_MODEL'))

#print("LOAD_MODEL " + str(LOAD_MODEL))
#if (LOAD_MODEL==1):
#  CHECKPOINT = os.getenv('CHECKPOINT')
#  print("CHECKPOINT " + str(CHECKPOINT))
#  try:
#    clf.load_weights(CHECKPOINT)
#    print('Model loaded\n')
#  except:
#    print("Problem loading checkpoint " + CHECKPOINT + "\n")
#    sys.exit(0)
  
#output_folder = './checkpoints/CustomNet_X160_Y160_' + dt_string
#if not os.path.exists(output_folder):
#  os.makedirs(output_folder)
##filepath=output_folder+"/model-{epoch:02d}-{val_iou_metric:.2f}.hdf5"
#filepath=output_folder+"/model-{epoch:02d}-{loss:.2f}-{val_loss:.2f}-{val_iou_metric:.2f}.hdf5"
#log_dir = 'tensorboard/tensorboard_' + dt_string
#os.makedirs('./'+log_dir, exist_ok=True)

callbacks = []

#callbacks = [
#  tf.keras.callbacks.ModelCheckpoint(
#    filepath=filepath,
#    #monitor='val_loss',
#    save_best_only=False
#  ), 
#  # tensorboard callback
#  tf.keras.callbacks.TensorBoard(
#    log_dir=log_dir,
#    histogram_freq=1
#  )
#]

params = {
  'batch_size': BATCH_SIZE,
  'shuffle': True,
  'data_path': data_path,
  'to_fit': True
}

train_generator = utils_training.DataGenerator (
  X_train,
  y_train,
  **params
)
  
val_params = {
  'batch_size': 1,
  'shuffle': False,
  'data_path': data_path,
  'to_fit': True
}

val_generator = utils_training.DataGenerator(
  X_val,
  y_val,
  **val_params
)

# Implementing genetic algorithm
from collections import namedtuple, defaultdict

SIZE = 5
NUM_GENERATIONS = 500
POP_SIZE = 10
OFFSPRING_SIZE = 10
CROSSOVER_PROB = 0.5

Individual = namedtuple('Individual', ['genome', 'fitness'])

X_train_data, y_train_data = train_generator.get_data()
X_val_data, y_val_data = val_generator.get_data()

X_train_data_tf = tf.constant(X_train_data)
y_train_data_tf = tf.constant(y_train_data)
X_val_data_tf = tf.constant(X_val_data)
y_val_data_tf = tf.constant(y_val_data)

def fitness(clf, NUM_EPOCHS, filepath=None):
    if os.path.exists(filepath):
        os.remove(filepath)
    if filepath != None:
        callbacks = [
          tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='accuracy',
            save_best_only=True
          )
        ]
    else:
        callbacks = []

    start = time.time()
    clf.fit(
      x = X_train_data_tf,
      y = y_train_data_tf,
      batch_size = BATCH_SIZE,
      epochs = NUM_EPOCHS,
      validation_data = (X_val_data_tf, y_val_data_tf),
      verbose=2,
      callbacks = callbacks
    )

    if filepath != None:
        clf.load_weights(filepath)

    loss, accuracy = clf.evaluate(
        X_val_data_tf,
        y_val_data_tf,
        batch_size = BATCH_SIZE,
        verbose = 2
    )
    end = time.time()
    print("TIME ELAPSED %f s" % (end-start))
    return accuracy

def new_individual():
    return utils_training.model_encaps(
        model_libs.CustomNet(
            ELAB_IMG_SIZE_X,
            ELAB_IMG_SIZE_Y
        )
    )

def mutation(parent):
    genome = utils_training.model_encaps(tf.keras.models.clone_model(parent.genome))
    genome.set_weights(parent.genome.get_weights())
    mut_layer = random.choice(genome.layers)
    weights = mut_layer.get_weights()
    new_weights = [np.random.random_sample(weight_part.shape) for weight_part in weights]
    mut_layer.set_weights(
        new_weights
    )
    return genome

def copy_parent(parent):
    genome = utils_training.model_encaps(tf.keras.models.clone_model(parent.genome))
    genome.set_weights(parent.genome.get_weights())
    return genome

report = list()

GPU_DEVICES = tf.config.list_physical_devices('GPU')
NUM_GPUS = len(GPU_DEVICES)

print("CREATING STARTING POPULATION")
population = [
    Individual(g, fitness(g, NUM_EPOCHS, "./checkpoint/partial.hdf5")) for g in [
        new_individual() for _ in range(POP_SIZE)
    ]
]

population = sorted(population, key=lambda i: i.fitness)
report.append(population[0].fitness)

threadLock = [threading.Lock() for i in range(NUM_GPUS)]
threadLockPop = threading.Lock()

class offspring_thread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
    def run(self):
        threadLock[self.threadID % NUM_GPUS].acquire()
        if (np.random.rand(1) < 0.2):
            parent = random.choice(population)
            genome = copy_parent(parent)
        else:
            parent = random.choice(population)
            genome = mutation(parent)
        filepath = "./checkpoint/partial%d.hdf5" % self.threadID
        fitness_ind = fitness(genome, NUM_EPOCHS, filepath)
        threadLockPop.acquire()
        population.append(Individual(genome=genome, fitness=fitness_ind))
        threadLockPop.release()
        threadLock[self.threadID % NUM_GPUS].release()

with open('perf_evolve.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow([str(i.fitness) for i in population])
    for generation in range(NUM_GENERATIONS):
        print("GENERATION %d" % generation)
        eval_threads = [offspring_thread(o, "Thread-" + str(o), o) for o in range(OFFSPRING_SIZE)]
        for o in range(OFFSPRING_SIZE):
            eval_threads[o].start()
        for o in range(OFFSPRING_SIZE):
            eval_threads[o].join()
        population = sorted(population, key=lambda i: i.fitness, reverse=True)[:POP_SIZE]
        report.append(population[0].fitness)
        csv_writer.writerow([str(i.fitness) for i in population])

# Run this in the command line before fit: 
# load_ext tensorboard


# Run this in the command line after fit: 
# tensorboard --logdir logs/fit

