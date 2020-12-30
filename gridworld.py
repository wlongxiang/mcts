import sys
from io import StringIO

from gym.envs.toy_text import discrete
import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridworldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.

    For example, a 4x4 grid looks as follows:

    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T

    x is your position and T are the two terminal states.

    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[9, 9]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')
        # TODO: can you start at a wall state??
        self.shape = shape

        nS = np.prod(shape)
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a: [] for a in range(nA)}

            # wall points from (2, 3..7), (3..6, 7), (8, 2..5)
            # these are corresponding to states according to s = 9*(r-1) + c-1
            self.wall_states = []
            # for g in [[[2], [3, 4,5,6,7]], [[3,4,5,6], [7]], [[8], [2,3,4,5]]]:
            for c in [3, 4, 5, 6, 7]:
                temp = 9 * (2 - 1) + c - 1
                self.wall_states.append(temp)
            for r in [3, 4, 5, 6]:
                temp = 9 * (r - 1) + 7 - 1
                self.wall_states.append(temp)
            for c in [2, 3, 4, 5]:
                temp = 9 * (8 - 1) + c - 1
                self.wall_states.append(temp)

            self.snake_pit_state = 9 * 6 + 5  # 59
            self.treasure_state = 9 * 8 + 8  # 80
            if s == self.snake_pit_state:
                reward = -50.0
            elif s == self.treasure_state:
                reward = 50.0
            else:
                reward = -1.0

            # We're stuck in a terminal state
            if self.is_terminal(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:
                # next_state_up = s if y == 0 else s - MAX_X
                if y == 0 or (y == 2 and x in [2, 3, 4, 5, 6]) or (y == 8 and x in [1, 2, 3, 4]):
                    # we are below the wall or below boundary
                    next_state_up = s
                else:
                    next_state_up = s - MAX_X

                # next_state_right = s if x == (MAX_X - 1) else s + 1

                if x == (MAX_X - 1) or (y == 1 and x == 1) or (y == 7 and x == 0) or (x == 5 and y in [2, 3, 4, 5]):
                    next_state_right = s
                else:
                    next_state_right = s + 1

                # next_state_down = s if y == (MAX_Y - 1) else s + MAX_X
                if y == (MAX_Y - 1) or (y == 0 and x in [2, 3, 4, 5, 6]) or (y == 7 and x in [1, 2, 3, 4]):
                    next_state_down = s
                else:
                    next_state_down = s + MAX_X

                # next_state_left = s if x == 0 else s - 1
                if x == 0 or (y == 7 and x == 5) or (x == 7 and y in [1, 2, 3, 4, 5]):
                    next_state_left = s
                else:
                    next_state_left = s - 1

                P[s][UP] = [(1.0, next_state_up, reward, self.is_terminal(next_state_up))]
                P[s][RIGHT] = [(1.0, next_state_right, reward, self.is_terminal(next_state_right))]
                P[s][DOWN] = [(1.0, next_state_down, reward, self.is_terminal(next_state_down))]
                P[s][LEFT] = [(1.0, next_state_left, reward, self.is_terminal(next_state_left))]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)

    def is_terminal(self, s):
        return True if s == self.snake_pit_state or s == self.treasure_state else False

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif self.is_terminal(s):
                output = " T "
            elif s in self.wall_states:
                output = " W "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()
