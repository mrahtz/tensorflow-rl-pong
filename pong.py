#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import gym

from IPython.core.debugger import Tracer

OBSERVATION_SIZE = 210 * 160
HIDDEN_LAYER_SIZE = 200

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

initial_action = env.action_space.sample()
action = initial_action

def process_reward(reward, state_action_pairs):
    states, actions = zip(*state_action_pairs)
    states = np.array(states)
    actions = np.array(actions)

    if reward == +1:
        desired_actions = actions
    elif reward == -1:
        desired_actions = np.logical_not(actions).astype(np.int32)

    sess.run(train_op, feed_dict={x: states, y: desired_actions})

for i_episode in range(2000):
    print("Episode", i_episode)
    observation = env.reset()
    last_observation = None

    grads_win = None
    grads_lose = None
    tvars = None # TODO get this from get_trainable_variables() instead
    action_dict = {2: 0, 3: 1}

    state_action_pairs = []

    for t in range(100000):
        #env.render()
        observation, reward, done, info = env.step(action)

        if reward != 0:
            process_reward(reward, state_action_pairs)
            state_action_pairs = []
            print("Reward %d" % reward)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

        if last_observation is not None:
            observation_delta = observation - last_observation
            last_observation = observation
        else:
            last_observation = observation
            continue

        # sum over all colour channels
        observation_delta = np.sum(observation_delta, axis=2)
        observation_delta = np.reshape(observation_delta, [-1])

        p = sess.run(p_op,
            feed_dict={x: np.reshape(observation_delta, [1, -1])})

        # p is probability of up
        if np.random.uniform() < p[0]:
            action = 2 # up
        else:
            action = 3 # down

        state_action_pairs.append((observation_delta, action_dict[action]))

    if not done:
        print("Maximum timesteps reached; resetting")
