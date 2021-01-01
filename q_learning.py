import sys
from collections import defaultdict
from random import choice
import random
import numpy as np

from gridworld import GridworldEnv


def q_learning(env, num_episodes=1000):
    # Start with an all 0 Q value function
    step_size = 0.8
    discount_factor = 1.0
    Q = defaultdict(dict)
    for _s in env.P:
        for _a in [0, 1, 2, 3]:
            Q[_s][_a] = random.uniform(0,1)
    for _ in range(num_episodes):
        # init_state = choice(list(set(env.P.keys()) - set(env.wall_states)))  # draw a random state to start, exc. wall
        init_state = choice(list(set(env.P.keys())))  # draw a random state to start
        # generate an episode
        while not env.is_terminal(init_state):
            init_action = max(Q[init_state].items(), key=lambda a: a[1])[0]  # choose the action with max Q value for state
            reward = env.P[init_state][init_action][0][2]
            next_state = env.P[init_state][init_action][0][1]
            # update Q value
            best_q_for_next_state = max(Q[next_state].items(), key=lambda a: a[1])[1]
            Q[init_state][init_action] += step_size * (reward + discount_factor * best_q_for_next_state
                                                       - Q[init_state][init_action])
            init_state = next_state
        env.plot_q_value(Q)
        sys.stdout.flush()
    return Q


if __name__ == '__main__':
    env = GridworldEnv((9, 9))
    print(env.P)
    env._render(mode="human")
    V = q_learning(env, 50000)
    print(V)
