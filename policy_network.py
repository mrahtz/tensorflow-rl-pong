import os.path
import numpy as np
import tensorflow as tf
from IPython.core.debugger import Tracer
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

DEBUG1 = False
DEBUG2 = False

class Network:
    def __init__(self, hidden_layer_size, learning_rate, checkpoints_dir):
        self.learning_rate = learning_rate

        self.sess = tf.InteractiveSession()

        self.observations = tf.placeholder(tf.float32,
                                           [None, 80, 80, 4])
        # +1 for up, -1 for down
        self.sampled_actions = tf.placeholder(tf.float32, [None, 1])
        self.advantage = tf.placeholder(
            tf.float32, [None, 1], name='advantage')

        x = tf.layers.conv2d(
                inputs=self.observations,
                filters=32,
                kernel_size=8,
                strides=4,
                activation=tf.nn.relu)

        x = tf.layers.conv2d(
                inputs=x,
                filters=64,
                kernel_size=4,
                strides=2,
                activation=tf.nn.relu)

        x = tf.layers.conv2d(
                inputs=x,
                filters=64,
                kernel_size=3,
                strides=1,
                activation=tf.nn.relu)

        w, h, f = x.shape[1:]
        x = tf.reshape(x, [-1, int(w * h * f)])

        x = tf.layers.dense(
                inputs=x,
                units=512,
                activation=tf.nn.relu)

        self.up_probability = tf.layers.dense(
            x,
            units=1,
            activation=tf.sigmoid,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        # Train based on the log probability of the sampled action.
        #
        # The idea is to encourage actions taken in rounds where the agent won,
        # and discourage actions in rounds where the agent lost.
        # More specifically, we want to increase the log probability of winning
        # actions, and decrease the log probability of losing actions.
        #
        # Which direction to push the log probability in is controlled by
        # 'advantage', which is the reward for each action in each round.
        # Positive reward pushes the log probability of chosen action up;
        # negative reward pushes the log probability of the chosen action down.
        self.loss = tf.losses.log_loss(
            labels=self.sampled_actions,
            predictions=self.up_probability,
            weights=self.advantage)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(checkpoints_dir,
                                            'policy_network.ckpt')

    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

    def forward_pass(self, frame_stack):
        frame_stack = np.array(frame_stack)
        # from 4x80x80 to 80x80x4
        frame_stack = np.moveaxis(frame_stack, source=0, destination=-1)
        global DEBUG1
        if DEBUG1:
            for i in reversed(range(4)):
                plt.figure()
                plt.imshow(frame_stack[:, :, i])
            plt.show()
        up_probability = self.sess.run(
            self.up_probability,
            feed_dict={self.observations: [frame_stack]})
        return up_probability

    def train(self, state_action_reward_tuples):
        global DEBUG2
        print("Training with %d (state, action, reward) tuples" %
              len(state_action_reward_tuples))

        states, actions, rewards = zip(*state_action_reward_tuples)
        states = np.array(states)
        states = np.moveaxis(states, source=1, destination=-1)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)

        if DEBUG2:
            for j in range(states.shape[0]):
                for i in reversed(range(4)):
                    plt.figure()
                    plt.imshow(states[j, :, :, i])
                plt.show()

        feed_dict = {
            self.observations: states,
            self.sampled_actions: actions,
            self.advantage: rewards
        }
        self.sess.run(self.train_op, feed_dict)
