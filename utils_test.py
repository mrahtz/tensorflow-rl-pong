import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from utils import EnvWrapper

class DummyEnv:
    def __init__(self):
        self.i = 0
        self.observation_space = None
        self.unwrapped = None

    def reset(self):
        o = np.zeros((210, 160, 3))
        return o

    def step(self, a):
        o = np.zeros((210, 160, 3))
        # Draw a horizontal series of marks
        draw_y = 10
        draw_x = 10
        while draw_x < 160:
            o[draw_y, draw_x, 0] = 255
            draw_x += 10
        # Draw a mark below the mark corresponding
        # to the current frame
        draw_y = 20
        draw_x = 10 + self.i * 10
        o[draw_y, draw_x, 0] = 255
        self.i += 1
        return o, 0, False, None

    def render(self):
        pass

def test(env):
    o = env.reset()
    for i in range(4):
        plt.figure()
        plt.title("Frame %d" % i)
        plt.imshow(o, cmap='gray')
        o = env.step(0)[0]
    plt.show()

def test_envwrapper():
    """
    Test EnvWrapper.
    """
    print("Frame 1 mark 1, frame 2 mark 2, frame 3 mark 3")
    env = EnvWrapper(DummyEnv(), pool=False, frameskip=1)
    test(env)
    print("Frame 1 mark 1, frame 2 mark 1,2, frame 3 mark 2,3")
    env = EnvWrapper(DummyEnv(), pool=True, frameskip=1)
    test(env)
    print("Frame 1 mark 2, frame 2 mark 4, frame 3 mark 6")
    env = EnvWrapper(DummyEnv(), pool=False, frameskip=2)
    test(env)
    print("Frame 1 mark 3, frame 2 mark 6, frame 3 mark 9")
    env = EnvWrapper(DummyEnv(), pool=False, frameskip=3)
    test(env)
    print("Frame 1 mark 2+3, frame 2 mark 5+6, frame 3 mark 8+9")
    env = EnvWrapper(DummyEnv(), pool=True, frameskip=3)
    test(env)

if __name__ == '__main__':
    test_envwrapper()
