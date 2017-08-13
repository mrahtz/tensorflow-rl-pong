import numpy as np
from skimage.color import rgb2yuv
from scipy.misc import imresize

# Based on Andrej's code
def karpathy_prepro(I):
    I = np.mean(I, axis=2)
    I = I / 255.0
    """ prepro 210x160 frame into 80x80 frame """
    I = I[34:194]  # crop
    I = I[::2, ::2]  # downsample by factor of 2
    I[I <= 0.4] = 0 # erase background
    I[I > 0.4] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float)

def pong_prepro(o):
    #o[:24] = 0  # black out score area
    o = rgb2yuv(o)[:, :, 0]  # extract Y (between 0 and 1)
    # mode='F': floating-point pixels
    o = imresize(o, (84, 84), interp='bilinear', mode='F')
    return o

def prepro_a(o):
    o = rgb2yuv(o)[:, :, 0]
    o = o[34:194]
    o = o[::2, ::2]  # downsample by factor of 2
    return o

def prepro_b(o):
    o = rgb2yuv(o)[:, :, 0]
    o = o[34:194]
    o = imresize(o, (84, 84), interp='bilinear', mode='F')
    return o

def prepro_c(I):
    I = np.mean(I, axis=2)
    I = I / 255.0
    I = I[34:194]  # crop
    I[I <= 0.4] = 0 # erase background
    I[I > 0.4] = 1 # everything else (paddles, ball) just set to 1
    I = imresize(I, (84, 84), interp='bilinear', mode='F')
    return I

def prepro_alpha(o):
    o = rgb2yuv(o)[:, :, 0]
    o = o[34:194]
    o = imresize(o, (84, 84), interp='bilinear', mode='F')
    return o

def prepro_beta(o):
    o = rgb2yuv(o)[:, :, 0]
    o = o[34:194]
    o[o < 0.4] = 0
    o = imresize(o, (84, 84), interp='bilinear', mode='F')
    return o

def prepro_gamma(o):
    o = rgb2yuv(o)[:, :, 0]
    o = o[34:194]
    o[o >= 0.4] = 1
    o = imresize(o, (84, 84), interp='bilinear', mode='F')
    return o

def prepro_delta(o):
    o = rgb2yuv(o)[:, :, 0]
    o = o[34:194]
    o[o < 0.4] = 0
    o[o >= 0.4] = 1
    o = imresize(o, (84, 84), interp='bilinear', mode='F')
    return o

class EnvWrapper():
    def __init__(self, env, pool=False, frameskip=1, prepro=None):
        self.env = env
        self.pool = pool
        self.prepro = prepro
        # 1 = don't skip
        # 2 = skip every other frame
        self.frameskip = frameskip
        self.action_space = env.action_space
        # gym.utils.play() wants these two
        self.observation_space = env.observation_space
        self.unwrapped = env.unwrapped

    def reset(self):
        o = self.env.reset()
        self.prev_o = o
        o = self.prepro(o)
        return o

    def step(self, a):
        i = 0
        done = False
        rs = []
        while i < self.frameskip and not done:
            o_raw, r, done, _ = self.env.step(a)
            rs.append(r)
            if not self.pool:
                o = o_raw
            else:
                # Note that the first frame to come out of this
                # _might_ be a little funny because the first prev_o
                # is the first frame after reset which at least in Pong
                # has a different colour palette (though it turns out
                # it works fine for Pong)
                o = np.maximum(o_raw, self.prev_o)
                self.prev_o = o_raw
            i += 1
        o = self.prepro(o)
        r = sum(rs)
        info = None
        return o, r, done, info

    def render(self):
        self.env.render()


def discount_rewards(rewards, discount_factor):
    discounted_rewards = np.zeros_like(rewards)
    for t in range(len(rewards)):
        discounted_reward_sum = 0
        discount = 1
        for k in range(t, len(rewards)):
            discounted_reward_sum += rewards[k] * discount
            discount *= discount_factor
            if rewards[k] != 0:
                # Don't count rewards from subsequent rounds
                break
        discounted_rewards[t] = discounted_reward_sum
    return discounted_rewards
