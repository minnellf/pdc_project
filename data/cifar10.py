import pickle
import sys
import numpy as np
import os
from subprocess import call

import pandas as pd
from skimage.io import imsave


def download(url):
    fname = os.path.basename(url)
    if not os.path.exists(fname):
        call('wget {}'.format(url), shell=True)

def convert(ids, X, out_img_folder='imgs'):
    for id_, x in zip(ids, X):
        imsave('{}/{}.png'.format(out_img_folder, id_), x)

def save_csv(ids, labels, out_csv):
    assert len(ids) == len(labels)
    cols = {
        'id' : ids,
        'class': labels,
    }
    pd.DataFrame(cols).to_csv(out_csv, index=False, columns=['id', 'class'])


def load_data():
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    download('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    call('tar xvf cifar-10-python.tar.gz -C .', shell=True)
    path = 'cifar-10-batches-py'
    
    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = _load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = _load_batch(fpath)

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def _load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = pickle.load(f)
    else:
        d = pickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels



if __name__ == '__main__':
    np.random.seed(42)
    (X_train, Y_train), (X_test, Y_test) = load_data()
    Y = np.concatenate((Y_train, Y_test), axis=0)
    ids = np.arange(0, len(X_train) + len(X_test))
    np.random.shuffle(ids)
    ids_train = ids[0:len(X_train)]
    ids_test = ids[len(X_train):]

    if not os.path.exists('imgs'):
        os.mkdir('imgs')
    
    convert(ids_train, X_train, out_img_folder='imgs')
    save_csv(ids_train, Y_train, out_csv='train.csv')

    convert(ids_test, X_test, out_img_folder='imgs')
    save_csv(ids_test, Y_test, out_csv='test.csv')

    save_csv(ids, Y, out_csv='full.csv')
