# TensorFlow reinforcement learning Pong agent

A Pong AI trained using policy gradients, implemented using TensorFlow and
OpenAI gym, based on Andrej Karpathy's [Deep Reinforcement Learning: Pong from
Pixels](http://karpathy.github.io/2016/05/31/rl/).

After 7,000 episodes of training, the result looks like:

![](images/playing.gif)

## Usage

First, install OpenAI Gym and TensorFlow.

Run without any arguments to train the AI from scratch. Checkpoints will be
saved every so often (see `--checkpoint_every_n_episodes`).
Run with `--load_checkpoint --render` to see how an AI trained on ~8,000 episode
plays.

### installing Gym
OpenAI Gym provides an easy-to-use suite of reinforcement learning tasks. To install Gym, you will need a Python environment setup. It's recommended to use Python 3.5 or later. Follow the steps below to install Gym:
* First, ensure that you have Python installed on your machine. You can download Python from [here](https://www.python.org/downloads/).
* Once Python is installed, open your terminal or command prompt.
* Use pip to install gym by running the following command:
``` bash
pip install gym
```

### Installing TensorFlow
TensorFlow is an open-source machine learning framework developed by Google. Here's how to install TensorFlow on your machine:
* Ensure that you have Python installed on your machine. TensorFlow supports Python 3.6 to 3.9.
* Open your terminal or command prompt.
* To install TensorFlow, run the following command:
  
``` bash
pip install tensorflow
```


## Understanding The Code

### pong.py

* Imports and Arguments: Necessary libraries are imported and command line arguments are set up to tweak hyperparameters.
* Constants: Constants for the actions UP and DOWN are defined along with a dictionary to map actions to policy network outputs.
* prepro Function: Preprocesses the 210x160x3 uint8 game frame into a simplified 80x80 1D float vector to reduce complexity.
* discount_rewards Function: Computes discounted rewards over a reward sequence to prioritize more immediate rewards.
* Environment Setup: Initializes the Gym environment for Pong and sets up the policy network.
* Training Loop: A continuous loop that represents the training process, where each iteration corresponds to one game episode.
* Rendering: If args.render is true, the game gets rendered to the screen.
* Policy Execution: The policy network predicts the probability of moving UP, a random action is sampled, the action is performed in the environment, and the state, action, reward tuple is recorded.
* Training Procedure: Every args.batch_size_episodes, the policy network is trained with the collected state, action, reward tuples.

### policy_network.py

* Network Class: The main policy network class which handles TensorFlow session, network architecture, saving/loading checkpoints, and training.
* Initialization: Sets up a simple two-layer neural network with ReLU activation, and a sigmoid output layer.
* forward_pass Method: Makes a forward pass through the network to get the probability of moving UP.
* Train Method: This method trains the network using a log loss function to encourage taking actions that result in winning and discourage actions that result in loss. Adam optimizer is used for minimizing loss.
* Checkpointing: Functions load_checkpoint and save_checkpoint are available to save and load training progress.


## Vocabulary

* 'Round': one match, in which one player gains a point
* 'Episode': a set of rounds that make up one game (usually around 20 or so -
  I'm not sure what logic the game uses to decide this)

## Training Time

* It takes about 500 episodes to see whether the agent is improving or not
* It takes about 7,000 episodes to get to a stage where the agent is winning
  half and losing half of the rounds

## Changes from Andrej's Code

* Andrej calculates gradients for each episode, accumulates them over a batch
  size of 10 episodes, and then applies them all in one go. I think this is
  based on a recommendation in [Asynchronous Methods for Deep Reinforcement
  Learning](https://arxiv.org/pdf/1602.01783.pdf). It looked like this was going
  to be a pain to do in TensorFlow, though, (see e.g.
  <http://stackoverflow.com/q/37710974>), so here we just use a batch size of
  one episode.
* Andrej uses RMSProp, but here we use Adam. (RMSProp wouldn't work - the AI
  would never improved - and I was never able to figure out why.)

## Lessons Learned

When you have a hypothesis that you want to test, think deliberately about what
the _cheapest_ way to test it is.

For example, for a while things weren't working, and while debugging I noticed
that Andrej's code initialises his RMSProp gradient history with zeros, while
TensorFlow initialises with ones. I hypothesised that this was a key factor, and
the test I came up with was to compile a custom version of TensorFlow with
RMSProp initialised using zeros. It later occurred to me that a much cheaper
test would have been to just change Andrej's code to initialise with ones
instead.

Acknowledging explicitly to yourself when you've got a hypothesis you want to
test rather than just randomly testing stuff out in a state of flow may help
with this.
