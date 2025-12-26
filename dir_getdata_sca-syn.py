# -*- coding: utf-8 -*-
import os
import os.path as osp
import sys
import numpy as np
import torch
import igraph as ig


SEED = 42
TEST_RATIO = 0.1     
NEG_RATIO  = 1       
AVERAGE_DEGREE = 5   
POWERS = range(10, 20)  
# ==========================

def _sample_test_neg(num_needed: int,
                     train_nodes_arr: np.ndarray,
                     pos_edge_set: set) -> np.ndarray:
    """
     train_nodes × train_nodes ，，。
     shape = (num_needed, 2)  np.ndarray[int64]
    """
    result = set()
    m = len(train_nodes_arr)
    
    while len(result) < num_needed:
        
        batch = max(2 * num_needed, 4 * (num_needed - len(result)))
        us = np.random.choice(train_nodes_arr, size=batch, replace=True)
        vs = np.random.choice(train_nodes_arr, size=batch, replace=True)
        mask = (us != vs)
        cand = np.stack([us[mask], vs[mask]], axis=1)

        
        for u, v in cand:
            tup = (int(u), int(v))
            if tup not in pos_edge_set and tup not in result:
                result.add(tup)
                if len(result) >= num_needed:
                    break
    neg = np.fromiter((x for t in result for x in t), dtype=np.int64)
    return neg.reshape(-1, 2)

def main():
    np.random.seed(SEED)

    for i in POWERS:
        num_node = 2 ** i
        p = AVERAGE_DEGREE / num_node

        
        print(f"Generating ER graph with n={num_node}, p={p:.8f} ...")
        sys.stdout.flush()
        G = ig.Graph.Erdos_Renyi(n=num_node, p=p, directed=True, loops=False)
        print("Generation done.")
        sys.stdout.flush()
        
        
        edges = np.array(G.get_edgelist(), dtype=np.int64)          # (E, 2)
        pos_edge_set = set(map(tuple, edges))

        
        perm = np.random.permutation(len(edges))
        train_pos_len = int(len(edges) * (1.0 - TEST_RATIO))
        train_pos = edges[perm[:train_pos_len]]
        test_pos_raw = edges[perm[train_pos_len:]]

        
        train_nodes = set(map(int, train_pos.flatten()))
        test_pos = np.array(
            [(u, v) for (u, v) in map(tuple, test_pos_raw)
             if (u in train_nodes) and (v in train_nodes)],
            dtype=np.int64
        )

        print(f"[n={num_node}] train_pos_len={len(train_pos)} "
              f"test_pos_raw={len(test_pos_raw)} test_pos_filtered={len(test_pos)}")
        sys.stdout.flush()
        
        
        test_neg_len = int(len(test_pos) * NEG_RATIO)
        train_nodes_arr = np.array(sorted(train_nodes), dtype=np.int64)
        test_neg = _sample_test_neg(test_neg_len, train_nodes_arr, pos_edge_set)

        
        
        train_data = torch.from_numpy(train_pos.T).long()      # (2, E_train)
        test_pos_t = torch.from_numpy(test_pos.T).long()       # (2, E_test_pos)
        test_neg_t = torch.from_numpy(test_neg.T).long()       # (2, E_test_neg)

        
        N = num_node
        test_deg = torch.bincount(test_pos_t[0], minlength=N)

        
        # transform_pt: train = {'edge': train_data.t()}, test = {'edge': test_pos.t(), 'edge_neg': test_neg.t()}
        train_dict = {'edge': train_data.t()}  # (E_train, 2)
        test_dict  = {'edge': test_pos_t.t(), 'edge_neg': test_neg_t.t()}  # (E,2)
        
        valid_dict = {
            'edge':      train_data[:, :2].t(),   
            'edge_neg':  train_data[:, :2].t()
        }

        
        savedir = f'.dir_datasets//ER_{num_node}_{AVERAGE_DEGREE}'
        if not osp.exists(savedir):
            os.makedirs(savedir)

        torch.save(
            {
                'train': train_dict,
                'test':  test_dict,
                'valid': valid_dict,
                'x':     None,          
                'remap': None,          
                'uniq_nodes': None,     
                'test_deg': test_deg    
            },
            osp.join(savedir, f'data.pt')
        )

        print(f"Saved: {osp.join(savedir, f'data.pt')} ; "
              f"Train E={train_dict['edge'].shape[0]}, "
              f"Test+={test_dict['edge'].shape[0]}, Test-={test_dict['edge_neg'].shape[0]}")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
