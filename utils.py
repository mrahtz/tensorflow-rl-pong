import numpy as np

# From Andrej's code
def prepro2(I):
    """ prepro 210x160x3 uint8 frame into 80x80 float matrix """
    I = I[34:194]  # crop
    I = I[::2, ::2, 1]  # downsample by factor of 2
    # 144
    I[I == 72] = 0  # erase background
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float)

def prepro(o):
    o = np.mean(o, axis=2)
    return o

class EnvWrapper():
    def __init__(self, env, pool=True, frameskip=4):
        self.env = env
        # gym.utils.play() wants these two
        self.observation_space = env.observation_space
        self.unwrapped = env.unwrapped
        self.pool = pool
        # 1 = don't skip
        # 2 = skip every other frame
        self.frameskip = frameskip

    def reset(self):
        o = self.env.reset()
        self.prev_o = o
        o = prepro(o)
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
        o = prepro(o)
        r = sum(rs)
        info = None
        return o, r, done, info

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
