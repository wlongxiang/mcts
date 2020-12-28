import random
from random import choice

from monte_carlo_tree_search import MCTS
import argparse


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


def make_binary_tree(depth=12):
    all_nodes = []
    # fix seed to make sure we get the same leaf values always
    # unseed in the end to enable MCTS has its exploration results random
    random.seed(123)
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
    random.seed()
    return root, leaf_nodes_dict


def main(args):
    mcts = MCTS(exploration_weight=args.exploration_weight)
    root, leaf_nodes_dict = make_binary_tree(depth=args.depth)
    leaf_nodes_dict_sorted = sorted(leaf_nodes_dict.items(), key=lambda x: x[1], reverse=True)
    print("Expected optimal (max) leaf node: {}, value: {}".format(leaf_nodes_dict_sorted[0][0],
                                                                   leaf_nodes_dict_sorted[0][1]))

    while True:
        # we run MCTS simulation for many times
        for _ in range(args.num_iter):
            mcts.run(root, num_rollout=args.num_rollout)
        # we choose the best greedy action based on simulation results
        root = mcts.choose(root)
        # we repeat until root is terminal
        if root.is_terminal():
            print("Found optimal (max) leaf node: {}, value: {}".format(root, root.value))
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCTS main runner')
    parser.add_argument("--num_iter", type=int, default=50,
                        help="number of MCTS iterations starting from a specific root node")
    parser.add_argument("--num_rollout", type=int, default=5, help="number of rollout simulations in a MCTS iteration")
    parser.add_argument("--depth", type=int, default=12, help="number of depth of the binary tree")
    parser.add_argument("--exploration_weight", type=float, default=1.0, help="exploration weight, c number in UCT")
    args = parser.parse_args()
    main(args)
