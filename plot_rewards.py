#!/usr/bin/env python3

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('rewards_file', type=argparse.FileType('rb'))
parser.add_argument('--moving_average_n', type=int, default=100)
args = parser.parse_args()

episode_reward_sums = pickle.load(args.rewards_file)

# http://stackoverflow.com/a/14314054
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

l = moving_average(episode_reward_sums, n=args.moving_average_n)
plt.plot(l)
plt.xlabel("Episode number (moving average)")
plt.ylabel("Reward sum over episode")
plt.show()
