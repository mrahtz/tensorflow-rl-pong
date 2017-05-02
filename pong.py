#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import tensorflow as tf
import gym

from IPython.core.debugger import Tracer

OBSERVATION_SIZE = 6400
HIDDEN_LAYER_SIZE = 200

UP_ACTION = 2
DOWN_ACTION = 3
action_dict = {DOWN_ACTION: 0, UP_ACTION: 1}

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, OBSERVATION_SIZE], name='x')
y = tf.placeholder(tf.float32, [None], name='y')

x1 = tf.layers.dense(x, units=HIDDEN_LAYER_SIZE)
x1 = tf.nn.relu(x1)

logp = tf.layers.dense(x1, units=1)
p_op = tf.sigmoid(logp)

loss_op = tf.reduce_mean(y - p_op)
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)
train_op = optimizer.minimize(loss_op)

tf.global_variables_initializer().run()

env = gym.make('Pong-v0')

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  """https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5"""
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()


def process_reward(reward, state_action_pairs):
    states, actions = zip(*state_action_pairs)
    states = np.array(states)
    actions = np.array(actions)

    if reward == +1:
        desired_actions = actions
    elif reward == -1:
        desired_actions = np.logical_not(actions).astype(np.int32)

    sess.run(train_op, feed_dict={x: states, y: desired_actions})

action = env.action_space.sample()
observation = env.reset()
last_observation = None
t = 0
game_n = 0
state_action_pairs = []

while True:
    #env.render()
    observation, reward, done, info = env.step(action)
    t += 1

    if reward != 0:
        print("Game %d: %d steps; " % (game_n, t), end='')
        if reward == +1:
            print("won!")
        else:
            print("lost!")

        if t > 100:
            reward = +1

        if reward == +1:
            process_reward(reward, state_action_pairs)

        state_action_pairs = []
        t = 0
        game_n += 1
        observation = env.reset()
        last_observation = None
        action = env.action_space.sample()

    observation = prepro(observation)

    if last_observation is not None:
        observation_delta = observation - last_observation
        last_observation = observation
    else:
        last_observation = observation
        continue

    up_probability = \
        sess.run(p_op, feed_dict={x: observation_delta.reshape([1, -1])})
    up_probability = up_probability[0]

    if np.random.uniform() < up_probability:
        action = UP_ACTION
    else:
        action = DOWN_ACTION

    state_action_pairs.append((observation_delta, action_dict[action]))
