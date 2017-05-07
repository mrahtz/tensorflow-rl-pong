# TensorFlow reinforcement learning Pong agent

A Pong agent trained using policy gradients, implemented using TensorFlow and
OpenAI gym.

Basically, an attempt at a TensorFlow version of Andrej Karpathy's
[Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/).

## Vocabulary

* 'Round': one match, in which one player gains a point
* 'Episode': a set of rounds (usually around 20 or so - I'm not sure what logic
  the game uses to decide this)

## Training Time

* It takes about 500 episodes to see whether the agent is improving or not
* It takes about 7,000 episodes to get to a stage where the agent is winning
  half and losing half of the rounds

After 7,000 episodes:

!(images/playing.gif)

## Changes from Andrej's Code

* Andrej accumulates gradients from each episode over a batch size of 10
  episodes, and then applies them all in one go. I think this is based on a
  recommendation in [Asynchronous Methods for Deep Reinforcement
  Learning](https://arxiv.org/pdf/1602.01783.pdf). It looked like this was going
  to be a pain to do in TensorFlow, though, (see e.g.
  <http://stackoverflow.com/q/37710974>), so here we just use a batch size of
  one episode.

## Lessons Learned

* Given that the number of rounds the agent wins and loses each episode jumps
  around so much, how can you track progress? Plot a moving average of the mean
  reward per episode. (500 episodes seems to be roughly enough to judge whether
  the agent is actually improving.)
* When you have a hypothesis that you want to test, think deliberately about
  what the _cheapest_ way to test it is.
  * For example, at some point things weren't working, and while debugging I
    noticed that Andrej's code initialises his RMSProp gradient history with
    zeros, while TensorFlow initialises with ones. I hypothesised that this was
    a key factor, and the test I came up with was to compile a custom version of
    TensorFlow with RMSProp initialised using zeros. It later occurred to me
    that a much cheaper test would have been to just change Andrej's code to
    initialise with ones instead.
  * Acknowledging explicitly to yourself when you've got a hypothesis you want
    to test rather than just randomly testing stuff out in a state of flow may
    help with this.
