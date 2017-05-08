import tensorflow as tf
import numpy as np

OBSERVATIONS_SIZE = 6400


class Network:
    def __init__(self, hidden_layer_size):
        self.sess = tf.InteractiveSession()

        self.observations = tf.placeholder(tf.float32,
                                           [None, OBSERVATIONS_SIZE])
        # +1 for up, -1 for down
        self.sampled_actions = tf.placeholder(tf.float32, [None, 1])
        self.advantage = tf.placeholder(
            tf.float32, [None, 1], name='advantage')

        x = tf.layers.dense(
            self.observations,
            units=hidden_layer_size,
            use_bias=False,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.nn.relu(x)

        x = tf.layers.dense(
            x,
            units=1,
            use_bias=False,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.up_probability = tf.sigmoid(x)

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
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001 / 2)
        self.train_op = optimizer.minimize(self.loss)

        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()

    def load_checkpoint(self, checkpoints_dir):
        print("Loading checkpoint...")
        self.saver.restore(self.sess, 'checkpoints/model.ckpt')

    def save_checkpoint(self, checkpoints_dir):
        self.saver.save(self.sess, 'checkpoints/model.ckpt')

    def forward_pass(self, observations):
        up_probability = self.sess.run(
            self.up_probability,
            feed_dict={self.observations: observations.reshape([1, -1])})
        up_probability = up_probability[0]
        return up_probability_val

    def train(self, state_action_reward_tuples):
        print("Training with %d (state, action, reward) tuples" %
              len(state_action_reward_tuples))

        states, actions, rewards = zip(*state_action_reward_tuples)
        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)

        feed_dict = {
            self.observations: states,
            self.sampled_actions: actions,
            self.advantage: rewards
        }
        self.sess.run(self.train_op, feed_dict)
