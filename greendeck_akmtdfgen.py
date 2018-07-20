"""
modifications on top of https://gist.github.com/timehaven/257eef5b0e2d9e2625a9eb812ca2226b
"""

from __future__ import print_function

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import numpy as np
import pandas as pd
import bcolz
import threading

import os
import sys
import glob
import shutil

from imgaug import augmenters as iaa

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical

seq = iaa.Sequential([
    iaa.Fliplr(0.5), #horizontally flip 50% of the images
    iaa.Affine(
        rotate=(-40,40),
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        shear=(-0.2, 0.2)
    )
], random_order = True)

bcolz_lock = threading.Lock()

def safe_bcolz_open(fname, idx=None, debug=False):
    with bcolz_lock:
        if idx is None:
            X2 = bcolz.open(fname)
        else:
            X2 = bcolz.open(fname)[idx]
    return X2


class threadsafe_iter(object):
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.

    https://github.com/fchollet/keras/issues/1638
    http://anandology.com/blog/using-iterators-and-generators/
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
        assert self.lock is not bcolz_lock

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.

    https://github.com/fchollet/keras/issues/1638
    http://anandology.com/blog/using-iterators-and-generators/
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator

def generator_from_df(df, df_overall, headings_dict, batch_size, target_size, features=None,
                      debug_merged=True, parametrization_dict = dict()):
    """Generator that yields (X, Y).

    If features is not None, assume it is the path to a bcolz array
    that can be indexed by the same indexing of the input df.

    Assume input DataFrame df has columns 'imgpath' and 'target', where
    'imgpath' is full path to image file.

    https://github.com/fchollet/keras/issues/1627
    https://github.com/fchollet/keras/issues/1638

    Be forewarned if/when you modify this function: some errors will
    not be explicit, appearing only as a generic:

      ValueError: output of generator should be a tuple `(x, y, sample_weight)` or `(x, y)`. Found: None

    It usually means something in your infinite loop is not doing what
    you think it is, so the loop crashes and returns None.  Check your
    DataFrame in this function with various print statements to see if
    it is doing what you think it is doing.

    Again, error messages will not be too helpful here--if in doubt,
    print().

    """
    if features is not None:
        assert os.path.exists(features)
        assert safe_bcolz_open(features).shape[0] == df.shape[0], "Features rows must match df!"

    # Each epoch will only process an integral number of batch_size
    # but with the shuffling of df at the top of each epoch, we will
    # see all training samples eventually, but will skip an amount
    # less than batch_size during each epoch.
    nbatches, n_skipped_per_epoch = divmod(df.shape[0], batch_size)

    # At the start of *each* epoch, this next print statement will
    # appear once for *each* worker specified in the call to
    # model.fit_generator(...,workers=nworkers,...)!
    #     print("""
    # Initialize generator:
    #   batch_size = %d
    #   nbatches = % 
    #   df.shape = %s
    # """ % (batch_size, nbatches, str(df.shape)))

    count = 1
    epoch = 0

    multi_class_array = parametrization_dict['multi_class']
    multi_label_array = parametrization_dict['multi_label']

    # getting binarizers
    label_binarizers = []
    for i in range(len(multi_class_array)):
        d = multi_class_array[i]
        column = None
        file = None
        for key in d:
            # column = df_overall.iloc[[d[key]]]
            column = df_overall.iloc[:,headings_dict[d[key]]:headings_dict[d[key]] + 1]
            file = open(key+'_classes.txt', 'w')
        column_np = np.array(column)
        lb = LabelBinarizer()
        lb.fit(column_np.astype(str))
        label_binarizers.append(lb)
        classes_arr = list(lb.classes_)
        for j in range(len(classes_arr)):
            if (i>0):
                file.write(',')
            file.write(classes_arr[j])
        file.close()
        

    multi_label_binarizers = []
    for i in range(len(multi_label_array)):
        d = multi_label_array[i]
        columns = None
        file = None
        for key in d:
            # columns = df_overall.iloc[d[key]]
            file = open(key+'_classes.txt', 'w')
            arr = d[key]
            dummy_arr = [] # indexes of required columns
            for element in arr:
                dummy_arr.append(headings_dict[element])
            columns = df_overall.iloc[:,dummy_arr[0]:dummy_arr[0]+1]
            for j in range(1, len(dummy_arr)):
                dummy_column = df_overall.iloc[:,dummy_arr[j]:dummy_arr[j]+1]
                columns = pd.concat([columns, dummy_column], axis = 1) # stacking horizontally
        columns_np = np.array(columns)
        mlb = MultiLabelBinarizer()
        mlb.fit(columns_np.astype(str))
        multi_label_binarizers.append(mlb)
        classes_arr = list(mlb.classes_)
        for j in range(len(classes_arr)):
            if (i>0):
                file.write(',')
            file.write(classes_arr[j])
        file.close()


    # New epoch.
    while 1:

        # The advantage of the DataFrame holding the image file name
        # and the labels is that the entire df fits into memory and
        # can be easily shuffled at the start of each epoch.
        #
        # Shuffle each epoch using the tricky pandas .sample() way.
        df = df.sample(frac=1)  # frac=1 is same as shuffling df.

        epoch += 1
        i, j = 0, batch_size

        # Mini-batches within epoch.
        mini_batches_completed = 0
        for _ in range(nbatches):

            # Callbacks are more elegant but this print statement is
            # included to be explicit.
            # print("Top of generator for loop, epoch / count / i / j = "\
            #       "%d / %d / %d / %d" % (epoch, count, i, j))

            sub = df.iloc[i:j]

            try:

                # preprocess_input()
                # https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py#L389
                z = np.array(sub.iloc[:,0:1])
                files = []
                for element in z:
                    files.append(element[0])
                X = np.array([

                        (2 *

                         # Resizing on the fly is efficient enough for
                         # pre-caching when a GPU is training a
                         # mini-batch.  Here is where some additional
                         # data augmentation could take place.
                         (img_to_array(load_img(f, target_size=target_size))

                          / 255.0 - 0.5))

                        for f in files])                
                # print (X.shape) 
                arr = [] # since imagaug requires an array instead of a numpy array?
                for element in X:
                    arr.append(element)

                arr = seq.augment_images(arr)

                X = np.array([element for element in arr])
                
                Y = dict()

                for index in range(len(multi_class_array)):
                    d = multi_class_array[index]
                    column = None
                    output_name = None
                    for key in d:
                        output_name = key
                        column = sub.iloc[:,headings_dict[d[key]]:headings_dict[d[key]] + 1].astype('str')
                    column_np = np.array(column)
                    column_labels = label_binarizers[index].transform(column_np)
                    Y[output_name] = column_labels

                for index in range(len(multi_label_array)):
                    d = multi_label_array[index]
                    columns = None
                    output_name = None
                    for key in d:
                        arr = d[key]
                        dummy_arr = [] # indexes of required columns
                        for element in arr:
                            dummy_arr.append(headings_dict[element])
                        columns = sub.iloc[:,dummy_arr[0]:dummy_arr[0]+1]
                        for jindex in range(1, len(dummy_arr)):
                            dummy_column = sub.iloc[:,dummy_arr[jindex]:dummy_arr[jindex]+1].astype('str')
                            columns = pd.concat([columns, dummy_column], axis = 1) # stacking horizontally                        
                        output_name = key
                    columns_np = np.array(columns)
                    columns_labels = multi_label_binarizers[index].transform(columns_np)
                    Y[output_name] = columns_labels

                if features is None:

                    # Simple model, one input, one output.
                    mini_batches_completed += 1
                    yield X, Y

                else:

                    # For merged model: two input, one output.
                    #
                    # HEY: You should probably test this very
                    # carefully!

                    # Make (slightly) more efficient by removing the
                    # debug_merged check.
                    X2 = safe_bcolz_open(features, sub.index.values, debug=debug_merged)

                    mini_batches_completed += 1

                    yield [X, X2], Y
                    # Or:
                    # yield [X, bcolz.open(features)[sub.index.values]], Y

            except IOError as err:

                # A type of lazy person's regularization: with
                # millions of images, if there are a few bad ones, no
                # need to find them, just skip their mini-batch if
                # they throw an error and move on to the next
                # mini-batch.  With the shuffling of the df at the top
                # of each epoch, the bad apples will be in a different
                # mini-batch next time around.  Yes, they will
                # probably crash that mini-batch, too, but so what?
                # This is easier than finding bad files each time.

                # Let's decrement count in anticipation of the
                # increment coming up--this one won't count, so to
                # speak.
                count -= 1

                # Actually, we could make this a try...except...else
                # with the count increment.  Homework assignment left
                # to the reader.

            i = j
            j += batch_size
            count += 1