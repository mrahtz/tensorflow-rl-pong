#!/usr/bin/env python

from __future__ import print_function

import argparse
import os.path
import pickle

import tensorflow as tf

from policy_network import Network
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('run_name')
parser.add_argument('prepro')
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--n')
args = parser.parse_args()

if args.prepro == 'c':
    prepro = prepro_c
elif args.prepro == 'b':
    prepro = prepro_b
elif args.prepro == 'alpha':
    prepro = prepro_alpha
elif args.prepro == 'beta':
    prepro = prepro_beta
elif args.prepro == 'gamma':
    prepro = prepro_gamma
elif args.prepro == 'delta':
    prepro = prepro_delta
else:
    raise Exception("Unsupported prepro: %s" % args.prepro)

if args.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer
elif args.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer
else:
    raise Exception("Unsupported optimizer: %s" % args.optimizer)

network = Network(args.learning_rate, checkpoints_dir='checkpoints',
        run_name=args.run_name, optimizer=optimizer)


print("Loading batch...")
with open('batch_1.pickle', 'rb') as f:
    l = pickle.load(f)
print("Done!")
print("Preprocessing...")
o, a, r = zip(*l)
o = o[:100]
a = a[:100]
r = r[:100]

print(r[:100])
from pylab import *
figure()
plot(r)
show()

o = list(o)
for o_n in range(len(o)):
    frames = []
    for frame_n in range(o[o_n].shape[0]):
        frames.append(prepro(o[o_n][frame_n]))
    o[o_n] = np.array(frames)
print("Done!")

l = list(zip(o, a, r))

if args.n is not None:
    max = int(args.n)
else:
    max = 10000
i = 1
while i < max:
    print("Loop", i)
    network.train(l, i)
    i += 1
