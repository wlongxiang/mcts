import random

from binary_tree import Node


def make_binary_tree_with_value(depth=12):
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

    for level, nodes in enumerate(all_nodes):
        for loc, n in enumerate(nodes):
            n.value = random.uniform(0, 100)
            # if it's not the last level, we attach its children
            if level < depth:
                left = all_nodes[level + 1][2 * loc]
                right = all_nodes[level + 1][2 * loc + 1]
                n.left = left
                n.right = right
    root = all_nodes[0][0]
    return root


def tree_search(root, mode="bfs"):
    """This implementation is according to the generic uninformed search method proposed in Peter & Russel 3.3"""
    # frontier is the set of all node available for expansion at any given point of time (set of unexpanded nodes)
    # init the frontier to contain the initial state, which is the root
    frontier = [root]
    while frontier:
        # FIFO to do BFS, and FILO to do DFS
        if mode == "bfs":
            leaf = frontier.pop(0)
        elif mode == "dfs":
            leaf = frontier.pop(-1)
        # check if the leaf satisfies goal state, if yes, we have found the solution
        print(leaf)
        # now we expand the leaf nodes, and its children (if any) are added to the frontier
        for child in leaf.find_children_list():
            frontier.append(child)


if __name__ == '__main__':
    root = make_binary_tree_with_value(3)
    tree_search(root)
