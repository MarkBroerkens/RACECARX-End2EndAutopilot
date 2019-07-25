"""Train the model"""
import logging
import os
import random

import tensorflow as tf

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import Params
from utils import set_logger
from utils import save_dict_to_json

from random import getrandbits

from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Cropping3D, Conv2DTranspose



def _parse_function(filename, labels, size):
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    image_string = tf.io.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image_decoded = tf.io.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    resized_image = tf.image.resize(image, [size, size])
    resized_image = tf.image.per_image_standardization(resized_image)

    return resized_image, labels


def _train_augment(image, labels):
    """Image preprocessing for training.

    Apply the following operations:
        - Horizontally flip the image with probability 1/2
        - Apply random brightness and saturation
    """
    if not getrandbits(1):
        image = tf.image.flip_left_right(image)
        # flip steering_angle
        labels = (labels[0] * -1,) + labels[1:]
    #image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    #image = tf.clip_by_value(image, 0.0, 1.0)

    return image, labels


def input_fn(is_training, filenames, labels, params):
    """Input function for the SIGNS dataset.

    The filenames have format "{label}_IMG_{id}.jpg".
    For instance: "data_dir/2_IMG_4584.jpg".

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.jpg"...]
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    parse_fn = lambda f, l: _parse_function(f, l, params.image_size)
    train_fn = lambda f, l: _train_augment(f, l)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((filenames, (labels[['steering_angle']], labels[['speed']])))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .repeat()
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((filenames, (labels[['steering_angle']], labels[['speed']])))
            .map(parse_fn)
            .repeat()
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    return dataset


def training_pipeline(train_filenames, train_labels, eval_filenames, eval_labels, model_dir, params):
    training_set = input_fn(True, train_filenames, train_labels, params)
    testing_set  = input_fn(False, eval_filenames, eval_labels, params)

    try:
        os.makedirs(model_dir)
    except OSError:
        if not os.path.isdir(model_dir):
            raise

    model_file = os.path.join(model_dir, "model.h5")

    # #############
    # Train Model
    # #############
    model = default_n_linear(num_outputs=2)  # your keras model here
    model.compile(optimizer='adam',
                  loss=['mean_squared_error', 'mean_squared_error'] #,
                  #metrics=['mean_squared_error','mean_squared_error']
                  )

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=params.min_delta,
                                               patience=params.early_stop_patience,
                                               verbose=1,
                                               mode='auto')

    save_best = keras.callbacks.ModelCheckpoint(model_file,
                                                monitor='val_loss',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=False,
                                                mode='auto')

    model.fit(
        training_set,
        steps_per_epoch=len(train_filenames) // params.batch_size,
        epochs=params.num_epochs,
        validation_data=testing_set,
        validation_steps=len(eval_filenames) // params.batch_size,
        verbose = 1,
        callbacks=[save_best, early_stop])


def adjust_input_shape(input_shape, roi_crop):
    height = input_shape[0]
    new_height = height - roi_crop[0] - roi_crop[1]
    return (new_height, input_shape[1], input_shape[2])


def default_n_linear(num_outputs=2, input_shape=(64, 64, 3), roi_crop=(0, 0)):
    drop = 0.1

    # we now expect that cropping done elsewhere. we will adjust our expeected image size here:
    input_shape = adjust_input_shape(input_shape, roi_crop)

    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu', name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu', name="conv2d_2")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu', name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu', name="conv2d_4")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu', name="conv2d_5")(x)
    x = Dropout(drop)(x)

    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(drop)(x)

    outputs = []

    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear', name='n_outputs' + str(i))(x))

    model = Model(inputs=[img_in], outputs=outputs)

    return model


def loadCSV(data_dir):
    # find the CSV file(s)
    dataframe_csv = os.path.join(data_dir, 'dataset.csv')
    dataframe = pd.read_csv(dataframe_csv)
    filenames = [os.path.join(data_dir, filename) for filename in dataframe.filename]

    labels = dataframe[['steering_angle', 'speed']]

    return filenames, labels


def train(data_dir, model_dir, params):
    set_logger(os.path.join(model_dir, 'train.log'))

    # find the CSV file(s)
    filenames, labels = loadCSV(data_dir)

    train_filenames, eval_filenames, train_labels, eval_labels = train_test_split(filenames, labels, test_size=0.2)

    training_pipeline(train_filenames, train_labels, eval_filenames, eval_labels, model_dir, params)
