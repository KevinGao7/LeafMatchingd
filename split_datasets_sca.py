import csv
import logging
import os
import os.path as osp
import sys
import random
sys.path.append("/home/gaochi/OurDataset/OurCogDL")

import numpy as np
import pandas as pd
import torch
from OurCogDL.cogdl.wrappers.data_wrapper.link_prediction.embedding_link_prediction_dw_sca import \
    EmbeddingLinkPredictionDataWrapper
from ogb.linkproppred import LinkPropPredDataset
import argparse
from typing import Tuple, Literal

np.random.seed(0)
random.seed(0)

from typing import Dict, Literal
import torch
import numpy as np
from torch_geometric.datasets import ICEWS18


def transform_pt(train_data: torch.Tensor, test_data: Tuple[torch.Tensor, torch.Tensor], node_feat=None):
    if train_data.shape[0] != 2 or test_data[0].shape[0] != 2 or test_data[1].shape[0] != 2:
        raise ValueError("Train and test data must have shape (2, E).")
    # if test_data[1].shape[1] != test_data[0].shape[1] * 5 and test_data[1].shape[1] != test_data[0].shape[1] * 1000:
    #     raise ValueError(f"Test negative edges must be 5 or 1000 times the positive edges. But we get neg = {test_data[1].shape[1]} v.s. pos = {test_data[0].shape[1]}.")
    
    train = {'edge': train_data.t()}
    test = {'edge': test_data[0].t(), 'edge_neg': test_data[1].t()}
    valid = {'edge': train_data[:, :2].t(), 'edge_neg': train_data[:, :2].t()}  # valid data is empty, just for compatibility
    
    logging.info(f"""Train Edges: {train['edge'].shape[1]}, 
                 Test Pos Edges: {test['edge'].shape[1]}, Test Neg Edges: {test['edge_neg'].shape[1]}""")
    
    return train, test, valid

def split_dataset(name, split_ratio, data_type):
    K = 5
    
    node_feat = None 
    assert name == 'twitter', "Only twitter dataset is supported in this script."
    
    dataset = np.load('datasets/twitter/edges.npy')
    node_feat = torch.from_numpy(np.load('datasets/twitter/features_top512.npy')).float()
    split_subdataset_path ='./datasets/twitter/split_node'
    
    if not osp.exists(split_subdataset_path):
        os.makedirs(split_subdataset_path)
    
    logging.info(f"Dataset: {name}, Split Subdataset Path: {split_subdataset_path}")
    
    datawrapper = EmbeddingLinkPredictionDataWrapper(
        dataset, negative_ratio=5, 
        icews18=('icews18' in name),
        twitter=(name == 'twitter'),
    )
    
    logging.info(f"Splitting ratio: {split_ratio}")

    # >>> Get the train/test split >>> #
    datawrapper.pre_transform(split_ratio)
    train_data: torch.Tensor = datawrapper.train_wrapper() # type: ignore
    test_data: Tuple[torch.Tensor, torch.Tensor] = datawrapper.test_wrapper() # type: ignore
    ### shape 都是 (2, E) 吗？形状检测：
    if train_data.shape[0] != 2 or test_data[0].shape[0] != 2 or test_data[1].shape[0] != 2:
        raise ValueError("Train and test data must have shape (2, E).")
    
    # >>> 手动重映射编号，因为 split 后某些节点不可见而无意义 <<< #
    def get_remap(train_data, test_data):
        all_nodes = torch.cat([train_data[0], train_data[1]]) # 我们保证了 test 的点都在 train 内
        uniq_nodes = torch.unique(all_nodes)
        remap = - torch.ones(torch.max(uniq_nodes).item() + 1, dtype=torch.long) # type: ignore
        for i, node in enumerate(uniq_nodes):
            remap[node] = i
        
        # 错误检测：test_data[0] 和 test_data[1] 中的节点是否都在 train_data 中
        if not torch.all(torch.isin(test_data[0], uniq_nodes)) and not torch.all(torch.isin(test_data[1], uniq_nodes)):
            raise ValueError("Test data contain nodes not in training data.")
        
        return remap, uniq_nodes
    
    remap, uniq_nodes = get_remap(train_data, test_data)
    train_data = remap[train_data] 
    test_data = (remap[test_data[0]], remap[test_data[1]])
    if node_feat is not None:
        node_feat = node_feat[uniq_nodes] 
        # uniq_nodes: 下标为新编号，值为旧编号; remap: 下标为旧编号，值为新编号; node_feat[uniq_nodes]: 对于 node_feat[i]，我们首先得到 uniq_nodes[i]，它代表第 i 个新编号对应的旧编号，所以 node_feat[uniq_nodes[i]] 代表的是第 i 个新编号对应的旧编号的 node_feat，所以正确。
    N = len(uniq_nodes)  # 节点总数
    test_deg = torch.bincount(test_data[0][0], minlength=N)  # 统计 test_pos 中每个节点的度数
    
    
    # >>> Transform and Save >>> #
    if data_type == 'pt':
        train, test, valid = transform_pt(train_data, test_data, node_feat=node_feat)
        torch.save(
            {'train': train, 'test': test, 'valid': valid, 'x': node_feat, 'remap': remap, 'uniq_nodes': uniq_nodes, 'test_deg': test_deg}, 
            osp.join(split_subdataset_path, f'split_dict_{split_ratio}.pt')
        )
        logging.info(f"""Train Edges: {train['edge'].shape[0]}, Test Pos Edges: {test['edge'].shape[0]}, Test Neg Edges: {test['edge_neg'].shape[0]}""") # type: ignore
        logging.info(f"""Train/Test Pos Edges: {train['edge'].shape[0] / test['edge'].shape[0]}""") # type: ignore
        logging.info(f"""Train Edges/Nodes: {len(train['edge'])} / {len(torch.unique(train['edge']))} 
                    = {len(train['edge']) / len(torch.unique(torch.unique(train['edge'])))}\n""")
    else:
        raise NotImplementedError("Only 'pt' data type is implemented in this script.")

    print("Everything is done, check the logs for details.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split Dataset')
    parser.add_argument('--split_ratio', type=float, default=0.02)
    parser.add_argument('--dataset', type=str, default='twitter',)
    parser.add_argument('--type', type=str, default='pt', choices=['pt', 'txt'])
    
    args = parser.parse_args()
    split_ratio = args.split_ratio
    dataset = args.dataset
    data_type = args.type
    
    logging.basicConfig(filename=f'./datasets/logs/split_{dataset}_scaling_ratio_{split_ratio}.log',
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    split_dataset(dataset, split_ratio, data_type)
    
    
    
