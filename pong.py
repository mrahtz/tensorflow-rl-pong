#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import tensorflow as tf
import gym

from IPython.core.debugger import Tracer

OBSERVATION_SIZE = 6400
HIDDEN_LAYER_SIZE = 200
BATCH_SIZE_EPISODES = 10
RENDER = False

UP_ACTION = 2
DOWN_ACTION = 3
action_dict = {DOWN_ACTION: 0, UP_ACTION: 1}

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, OBSERVATION_SIZE], name='x')
y = tf.placeholder(tf.float32, [None], name='y')
advantage = tf.placeholder(tf.float32, [None], name='advantage')

x1 = tf.layers.dense(x, units=HIDDEN_LAYER_SIZE,
        kernel_initializer=tf.contrib.layers.xavier_initializer())
x1 = tf.nn.relu(x1)

logp = tf.layers.dense(x1, units=1,
        kernel_initializer=tf.contrib.layers.xavier_initializer())
p_op = tf.sigmoid(logp)

batch_losses = -1 * advantage * (y * tf.log(p_op) + (1 - y) * tf.log(1 - p_op))
loss_op = tf.reduce_mean(batch_losses)
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)
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


def train(state_action_reward_tuples):
    print("Training based on %d (state, action, reward) tuples" %
            len(state_action_reward_tuples))

    states, actions, rewards = zip(*state_action_reward_tuples)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)

    sess.run(train_op, feed_dict={x: states, y: actions, advantage: rewards})


last_observation = None
episode_n = 1

while True:
    print("Episode %d" % episode_n)
    env.reset()
    action = env.action_space.sample()
    state_action_tuples = []
    state_action_reward_tuples = []
    done = False

    game_n = 1
    t = 0
    while not done:
        if RENDER:
            env.render()

        observation, reward, done, info = env.step(action)
        observation = prepro(observation)
        t += 1

        if reward == -1:
            print("Game %d: %d time steps; lost..." % (game_n, t))
        elif reward == +1:
            print("Game %d: %d time steps; won!" % (game_n, t))
        if reward != 0:
            game_n += 1
            t = 0

            states, actions = zip(*state_action_tuples)
            assert(len(states) == len(actions))
            rewards = len(states) * [reward]
            assert(len(states) == len(rewards))
            state_action_reward_tuples.extend(zip(states, actions, rewards))
            state_action_tuples = []

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

        state_action_tuples.append((observation_delta, action_dict[action]))

    print("Episode %d finished after %d games" % (episode_n, game_n))
    if episode_n % BATCH_SIZE_EPISODES == 0:
        train(state_action_reward_tuples)
    episode_n += 1
