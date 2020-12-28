import random
from random import choice

from monte_carlo_tree_search import MCTS


class Node:
    def __init__(self, node_id=None):
        self.left = None
        self.right = None
        # assign a random value in [0. 100] to the end nodes as reward
        self.node_id = node_id
        self.value = None

    def __repr__(self):
        return "node@" + str(self.node_id)

    def __str__(self):
        return "node@" + str(self.node_id)

    def is_terminal(self):
        if self.left is None and self.right is None:
            return True
        return False

    def find_children(self):
        if self.is_terminal():
            return {}
        return {self.left, self.right}

    def find_random_child(self):
        if self.is_terminal():
            return None
        if choice([0, 1]) == 0:
            return self.left
        else:
            return self.right

    def reward(self):
        return self.value


def make_bintree(depth=12):
    all_nodes = []
    for i in range(depth + 1):
        nodes_at_depth = []
        num_of_nodes = pow(2, i)
        for j in range(num_of_nodes):
            nodes_at_depth.append(Node(str(i) + "_" + str(j)))
        all_nodes.append(nodes_at_depth)

    leaf_nodes_dict = dict()
    for level, nodes in enumerate(all_nodes):
        for loc, n in enumerate(nodes):
            if level >= len(all_nodes) - 1:
                # we assign reward value to leaf nodes of the tree
                n.value = random.uniform(0, 100)
                leaf_nodes_dict[n] = n.value
            else:
                left = all_nodes[level + 1][2 * loc]
                right = all_nodes[level + 1][2 * loc + 1]
                n.left = left
                n.right = right
    root = all_nodes[0][0]
    return root, leaf_nodes_dict


def play_game(num_sim):
    mcts = MCTS(exploration_weight=1)
    root, leaf_nodes_dict = make_bintree(depth=12)
    leaf_nodes_dict_sorted = sorted(leaf_nodes_dict.items(), key=lambda x: x[1], reverse=True)
    print("top 3 leaf nodes:", leaf_nodes_dict_sorted[:3])
    while True:
        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        for _ in range(num_sim):
            mcts.do_rollout(root)
        root = mcts.choose(root)
        if root.is_terminal():
            print("found optimal leaf node: {}, value: {}".format(root, root.value))
            break


if __name__ == '__main__':
    play_game(100)
