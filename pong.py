#!/usr/bin/env python
"""
Train a Pong AI using policy gradient-based reinforcement learning.

Based on Andrej Karpathy's "Deep Reinforcement Learning: Pong from Pixels"
http://karpathy.github.io/2016/05/31/rl/
and the associated code
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
"""

from __future__ import print_function

import argparse
import pickle
import numpy as np
import gym
import time

import os
print("Importing Tensorflow...")
import tensorflow as tf
print("Done!")

from policy_network import Network
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--batch_size_episodes', type=int, default=1)
parser.add_argument('--checkpoint_every_n_episodes', type=int, default=10)
parser.add_argument('--load_checkpoint', action='store_true')
parser.add_argument('--discount_factor', type=int, default=0.99)
parser.add_argument('--render', action='store_true')
args = parser.parse_args()

# Action values to send to gym environment to move paddle up/down
NO_ACTION = 1
UP_ACTION = 2
DOWN_ACTION = 3
ACTIONS = [NO_ACTION, UP_ACTION, DOWN_ACTION]

N_MAX_NOOPS = 30

print("Initialising...")

env = EnvWrapper(gym.make('PongNoFrameskip-v4'), prepro=prepro_b, frameskip=4)

network = Network(
    args.learning_rate, checkpoints_dir='checkpoints')
if args.load_checkpoint:
    network.load_checkpoint()

batch_state_action_reward_tuples = []
episode_n = 1

reward_average = None
reward_average_var = tf.Variable(0.0)
reward_summary = tf.summary.scalar('reward_average', reward_average_var)
dirname = 'summaries/' + str(int(time.time()))
os.makedirs(dirname)
summary_writer = tf.summary.FileWriter(dirname, flush_secs=1)

def log_rewards(reward_sum, step):
    global reward_average, reward_average_var, reward_summary, network, summary_writer
    if reward_average is None:
        reward_average = reward_sum
    else:
        reward_average = reward_average * 0.99 + reward_sum * 0.01
    print("Reward total was %.3f; reward average is %.3f" % (reward_sum,
          reward_average))
    network.sess.run(tf.assign(reward_average_var, reward_average))
    summ = network.sess.run(reward_summary)
    summary_writer.add_summary(summ, step)

print("Done!")

while True:
    print("Starting episode %d" % episode_n)

    episode_done = False
    episode_reward_sum = 0
    frame_stack = []

    env.reset()

    print("Noops...")
    n_noops = np.random.randint(low=0, high=N_MAX_NOOPS+1)
    for i in range(n_noops):
        env.step(0)
        if args.render:
            env.render()
    print("Done")

    for i in range(4):
        o, _, _, _ = env.step(0) # do nothing
        frame_stack.append(o)

    round_n = 1
    n_steps = 1

    while not episode_done:
        if args.render:
            env.render()

        a_p = network.forward_pass(frame_stack)[0]
        action = np.random.choice(ACTIONS, p=a_p)

        observation, reward, episode_done, _ = env.step(action)
        episode_reward_sum += reward
        n_steps += 1

        tup = (np.copy(frame_stack), ACTIONS.index(action), reward)
        batch_state_action_reward_tuples.append(tup)

        # NB this needs to happen _after_ we've recorded the last frame_stack
        frame_stack[:-1] = frame_stack[1:]
        frame_stack[-1] = observation

        if reward == -1:
            print("Round %d: %d time steps; lost..." % (round_n, n_steps))
        elif reward == +1:
            print("Round %d: %d time steps; won!" % (round_n, n_steps))
        if reward != 0:
            round_n += 1
            n_steps = 0

    print("Episode %d finished after %d rounds" % (episode_n, round_n))

    log_rewards(episode_reward_sum, episode_n)

    if episode_n % args.batch_size_episodes == 0:
        states, actions, rewards = zip(*batch_state_action_reward_tuples)
        rewards = discount_rewards(rewards, args.discount_factor)
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards)
        batch_state_action_reward_tuples = list(zip(states, actions, rewards))
        network.train(batch_state_action_reward_tuples)
        batch_state_action_reward_tuples = []

    if episode_n % args.checkpoint_every_n_episodes == 0:
        network.save_checkpoint()

    episode_n += 1
