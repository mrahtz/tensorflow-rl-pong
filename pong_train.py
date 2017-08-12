#!/usr/bin/env python

from __future__ import print_function

import argparse
import os.path
import pickle

from policy_network import Network
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.0001)
args = parser.parse_args()

network = Network(args.learning_rate, checkpoints_dir='checkpoints')

i = 1
while os.path.exists('batch_%d.pickle' % i):
    print(i)
    print("Loading batch...")
    with open('batch_%d.pickle' % i, 'rb') as f:
        l = pickle.load(f)
    print("Done!")
    print("Preprocessing...")
    o, a, r = zip(*l)
    o = list(o)
    for o_n in range(len(o)):
        frames = []
        for frame_n in range(o[o_n].shape[0]):
            frames.append(prepro(o[o_n][frame_n]))
        o[o_n] = np.array(frames)
    print("Done!")

    l = list(zip(o, a, r))
    network.train(l)
