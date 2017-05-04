#!/usr/bin/env python

from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf
import gym

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_layer_size', type=int, default=200)
parser.add_argument('--batch_size_episodes', type=int, default=10)
parser.add_argument('--render', action='store_true')
args = parser.parse_args()

OBSERVATION_SIZE = 6400
# Action values to send to environment to move paddle up/down
UP_ACTION = 2
DOWN_ACTION = 3
# Mapping from action values to outputs from the policy network
action_dict = {DOWN_ACTION: 0, UP_ACTION: 1}


def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  # From Karpathy's code:
  # https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()


def train(state_action_reward_tuples):
    print("Training with %d (state, action, reward) tuples" %
            len(state_action_reward_tuples))

    states, actions, rewards = zip(*state_action_reward_tuples)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)

    sess.run(train_op, feed_dict={observations: states,
                                  sampled_actions: actions,
                                  advantage: rewards})


# TensorFlow setup

sess = tf.InteractiveSession()
observations = tf.placeholder(tf.float32, [None, OBSERVATION_SIZE])
# +1 for up, -1 for down
sampled_actions = tf.placeholder(tf.float32, [None])
advantage = tf.placeholder(tf.float32, [None], name='advantage')

x = tf.layers.dense(observations, units=args.hidden_layer_size,
        kernel_initializer=tf.contrib.layers.xavier_initializer())
x = tf.nn.relu(x)

x = tf.layers.dense(x, units=1,
        kernel_initializer=tf.contrib.layers.xavier_initializer())
up_probability = tf.sigmoid(x)

# Train based on the log probability of the sampled action.
#
# If the sampled action was part of a round in which the agent won, we assume
# the sampled action was a good one, so we try to maximise the log probability
# of that action in the future. Here, therefore, 'advantage' is positive.
#
# If the sampled action was part of a round in which the agent didn't win, we
# assume the sampled action was a bad choice, so in that case we want to
# _minimise_ the log probability of that action in the future. 'advantage' is
# therefore negative.
#
# (The '-1' at the end is necessary because with TensorFlow we can only minimise
# a loss, whereas we actually want to maximise this quantity. For e.g. a good
# sampled action, we want to maximise the log probability, and this is the same
# as minimising the negative log probability.)
batch_losses = \
        (
            sampled_actions       * tf.log(up_probability) + \
            (1 - sampled_actions) * tf.log(1 - up_probability) \
        ) \
        * advantage \
        * -1
loss = tf.reduce_mean(batch_losses)
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)
train_op = optimizer.minimize(loss)

tf.global_variables_initializer().run()

# Set up OpenAI gym environment

env = gym.make('Pong-v0')

# Start training!

batch_state_action_reward_tuples = []
running_reward = None
episode_n = 1

while True:
    print("Starting episode %d" % episode_n)

    episode_done = False
    episode_reward_sum = 0

    round_state_action_tuples = []
    round_n = 1

    last_observation = env.reset()
    last_observation = prepro(last_observation)
    action = env.action_space.sample()
    observation, _, _, _ = env.step(action)
    observation = prepro(observation)
    n_steps = 1

    while not episode_done:
        if args.render:
            env.render()

        observation_delta = observation - last_observation
        last_observation = observation
        up_probability_val = sess.run(up_probability,
            feed_dict={observations: observation_delta.reshape([1, -1])})
        up_probability_val = up_probability_val[0]
        if np.random.uniform() < up_probability_val:
            action = UP_ACTION
        else:
            action = DOWN_ACTION
        tups = (observation_delta, action_dict[action])
        round_state_action_tuples.append(tups)

        observation, reward, episode_done, info = env.step(action)
        observation = prepro(observation)
        episode_reward_sum += reward
        n_steps += 1

        if reward == -1:
            print("Round %d: %d time steps; lost..." % (round_n, n_steps))
        elif reward == +1:
            print("Round %d: %d time steps; won!" % (round_n, n_steps))
        if reward != 0:
            # End of round
            round_n += 1
            n_steps = 0

            states, actions = zip(*round_state_action_tuples)
            rewards = len(states) * [reward]
            tups = zip(states, actions, rewards)
            batch_state_action_reward_tuples.extend(tups)
            round_state_action_tuples = []

    print("Episode %d finished after %d rounds" % (episode_n, round_n))

    # From Karpathy's code
    # https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
    # to enable comparison
    if running_reward is None:
        running_reward = episode_reward_sum
    else:
        running_reward = running_reward * 0.99 + episode_reward_sum * 0.01
    print("Reward total was %.3f; moving average of reward is %.3f" \
        % (episode_reward_sum, running_reward))
    episode_reward_sum = 0

    if episode_n % args.batch_size_episodes == 0:
        train(batch_state_action_reward_tuples)
        batch_state_action_reward_tuples = []
    episode_n += 1
