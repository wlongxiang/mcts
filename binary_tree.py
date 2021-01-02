from random import choice
import random


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

    def find_children_list(self):
        """This makes sure left, right order instead of an unsorted set"""
        if self.is_terminal():
            return []
        return [self.left, self.right]


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
    # random.seed(123)
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
    # random.seed()
    return root, leaf_nodes_dict
