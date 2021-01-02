from binary_tree import make_binary_tree
from monte_carlo_tree_search import MCTS
import argparse


def mcts_playout(depth, num_iter, num_rollout, exploration_weight):
    root, leaf_nodes_dict = make_binary_tree(depth=depth)
    leaf_nodes_dict_sorted = sorted(leaf_nodes_dict.items(), key=lambda x: x[1], reverse=True)
    print("Expected (max) leaf node: {}, value: {}".format(leaf_nodes_dict_sorted[0][0],
                                                           leaf_nodes_dict_sorted[0][1]))
    print("Expected (min) leaf node: {}, value: {}".format(leaf_nodes_dict_sorted[-1][0],
                                                           leaf_nodes_dict_sorted[-1][1]))

    mcts = MCTS(exploration_weight=exploration_weight)
    while True:
        # we run MCTS simulation for many times
        for _ in range(num_iter):
            mcts.run(root, num_rollout=num_rollout)
        # we choose the best greedy action based on simulation results
        root = mcts.choose(root)
        # we repeat until root is terminal
        if root.is_terminal():
            print("Found optimal (max) leaf node: {}, value: {}".format(root, root.value))
            return root.value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCTS main runner')
    parser.add_argument("--num_iter", type=int, default=50,
                        help="number of MCTS iterations starting from a specific root node")
    parser.add_argument("--num_rollout", type=int, default=1, help="number of rollout simulations in a MCTS iteration")
    parser.add_argument("--depth", type=int, default=12, help="number of depth of the binary tree")
    parser.add_argument("--exploration_weight", type=float, default=51, help="exploration weight, c number in UCT")
    args = parser.parse_args()
    mcts_playout(args.depth, args.num_iter, args.num_rollout, args.exploration_weight)
