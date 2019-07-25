import autopilot.train as ap
from autopilot.utils import Params
from autopilot.utils import set_logger

import argparse
import logging
import os
import random

import tensorflow as tf

import pandas as pd



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='experiments/learning_rate',
                        help="Experiment directory containing params.json")
    parser.add_argument('--data_dir', default='/data/cleaned',
                        help="Directory containing the dataset")
    parser.add_argument('--restore_from', default=None,
                        help="Optional, directory or file containing weights to reload before training")

    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    ap.train(args.data_dir, args.model_dir, params)


if __name__ == '__main__':
    print(tf.__version__)
    main()
