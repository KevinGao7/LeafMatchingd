import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


SEED = 42
np.random.seed(SEED)
for i in range(10, 20):

    num_node = 2**i
    test_ratio = 0.1
    neg_ratio = 1
    average_degree = 5
    p = average_degree/num_node

    # positive edges
    G = nx.erdos_renyi_graph(num_node, p, seed=SEED)
    pos_edge_set = set(G.edges())
    pos_edge_array = np.array(list(pos_edge_set))

        # split into train/test
    perm_indices = np.random.permutation(len(pos_edge_array))
    train_pos_len = int(len(pos_edge_array) * (1 - test_ratio))
    train_pos = pos_edge_array[perm_indices[:train_pos_len]]
    test_pos = pos_edge_array[perm_indices[train_pos_len:]]

        # filter out unobserved edges
    train_nodes = set(train_pos.flatten())
    test_pos_filtered = [edge for edge in test_pos if edge[0] in train_nodes and edge[1] in train_nodes]
    print(train_pos_len, len(test_pos), len(test_pos_filtered))
    test_pos = np.array(test_pos_filtered)

    # negative edges
    test_neg_len = int(len(test_pos) * neg_ratio)
    test_neg = set()
    while len(test_neg) < test_neg_len:
        new_edge = (np.random.randint(0, num_node), np.random.randint(0, num_node))
        if new_edge not in pos_edge_set and new_edge[0] in train_nodes and new_edge[1] in train_nodes:
            test_neg.add(new_edge)

    # save
    savedir = f'.undir_datasets/ER_{num_node}_{average_degree}/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    train_pos_df = pd.DataFrame(train_pos)
    train_pos_df.to_csv(f'{savedir}/train_pos.txt', index=False, header=False, sep='\t')
    test_pos_df = pd.DataFrame(test_pos)
    test_pos_df.to_csv(f'{savedir}/test_pos.txt', index=False, header=False, sep='\t')
    test_neg_df = pd.DataFrame(list(test_neg))
    test_neg_df.to_csv(f'{savedir}/test_neg.txt', index=False, header=False, sep='\t')

print('done')