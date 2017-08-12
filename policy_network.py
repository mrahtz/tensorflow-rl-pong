import os.path
import numpy as np
import tensorflow as tf
#import matplotlib
#matplotlib.use('Qt5Agg')
#import matplotlib.pyplot as plt

DEBUG1 = False
DEBUG2 = False

N_ACTIONS = 3

class Network:
    def __init__(self, learning_rate, checkpoints_dir):
        self.learning_rate = learning_rate

        self.sess = tf.InteractiveSession()

        self.observations = tf.placeholder(tf.float32,
                                           [None, 84, 84, 4])
        self.sampled_actions = tf.placeholder(tf.float32, [None])
        self.advantage = tf.placeholder(
            tf.float32, [None], name='advantage')

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

        a_logits = tf.layers.dense(
        inputs=x,
        units=N_ACTIONS,
        activation=None)

        self.a_softmax = tf.nn.softmax(a_logits)

        p = 0
        for i in range(N_ACTIONS):
            p += tf.cast(tf.equal(self.sampled_actions, i), tf.float32) \
                 * self.a_softmax[:, i]

        # Log probability: higher is better for actions we want to encourage
        # Negative log probability: lower is better for actions we want to
        # encourage
        # 1e-7: prevent log(0)
        nlp = -1 * tf.log(p + 1e-7)

        self.loss = tf.reduce_mean(nlp * self.advantage)
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
        # from 4x84x84 to 84x84x4
        frame_stack = np.moveaxis(frame_stack, source=0, destination=-1)
        global DEBUG1
        if DEBUG1:
            for i in reversed(range(4)):
                plt.figure()
                plt.imshow(frame_stack[:, :, i])
            plt.show()
        a_p = self.sess.run(
            self.a_softmax,
            feed_dict={self.observations: [frame_stack]})
        return a_p

    def train(self, state_action_reward_tuples):
        global DEBUG2
        print("Training with %d (state, action, reward) tuples" %
              len(state_action_reward_tuples))

        states, actions, rewards = zip(*state_action_reward_tuples)
        states = np.array(states)
        states = np.moveaxis(states, source=1, destination=-1)
        actions = np.vstack(actions).flatten()
        rewards = np.vstack(rewards).flatten()

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
