from collections import defaultdict
from random import choice

import numpy as np

from gridworld import GridworldEnv


def mc_policy_evaluation_random_policy(env, num_episodes=1000):
    # Start with an all 0 value function
    V = defaultdict(float)
    returns = defaultdict(list)  # an empty list for each state
    for i in range(num_episodes):
        episodes = []
        # init_state = choice(list(set(env.P.keys()) - set(env.wall_states)))  # draw a random state to start, exc. wall
        init_state = choice(list(set(env.P.keys())))  # draw a random state to start, exc. wall
        # generate an episode
        while True:

            action = choice(list(env.P[init_state].keys()))  # random policy such that draw an action randomly
            next_state = env.P[init_state][action][0][1]
            reward = env.P[init_state][action][0][2]
            episodes.append([init_state, action, reward])
            if env.is_terminal(init_state):
                break
            init_state = next_state
        G = 0
        states_seen = set()
        for S, A, R in reversed(episodes):
            G = 1.0 * G + R  # assuming discount factor is 1.0
            if S not in states_seen:
                states_seen.add(S)
                returns[S].append(G)
                V[S] = np.mean(returns[S])
    # terminal states have return 0 always
    V[env.treasure_state] = 0.0
    V[env.snake_pit_state] = 0.0

    assert len(V) == env.nS  # at least each state needs to be at least estimated once
    V_sorted = sorted(V.items(), key=lambda x: x[0])  # sort by state
    return V_sorted


if __name__ == '__main__':
    env = GridworldEnv((9, 9))
    print(env.P)
    env._render(mode="human")
    V = mc_policy_evaluation_random_policy(env, 5000)
    print(V)
