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
from OurCogDL.cogdl.wrappers.data_wrapper.link_prediction.embedding_link_prediction_dw_dir import \
    EmbeddingLinkPredictionDataWrapper
from OurCogDL.cogdl.wrappers.data_wrapper.link_prediction.embedding_link_prediction_dw_dir_cit import \
    EmbeddingLinkPredictionDataWrapper_cit2
from torch_geometric.datasets import ICEWS18
from torch_geometric.datasets import CitationFull
from ogb.linkproppred import LinkPropPredDataset
import argparse
from typing import Tuple, Literal

np.random.seed(0)
random.seed(0)

def check_citation2_data(test_pos: torch.Tensor, test_neg: torch.Tensor,
                         test_deg: torch.Tensor, N: int, K: int):

    # ---------- active nodes & K ----------
    active_u = torch.nonzero(test_deg > 0, as_tuple=False).flatten()  # [A]
    A = active_u.numel()
    total_neg = test_neg.size(1)
    assert total_neg % A == 0, "test_neg length must be A*K"
    assert total_neg // A == K, "test_neg length must be A*K"
    K = total_neg // A

    # ---------- shape & ordering checks ----------
    assert test_neg.shape == (2, A * K), "test_neg shape mismatch"
    assert test_pos.shape[0] == 2, "test_pos must be [2,E]"

    # (1) test_neg row-0 must be u repeated K times, ascending u
    u_in_neg = test_neg[0].view(A, K)[:, 0]
    assert torch.equal(u_in_neg, active_u), \
            "test_neg row-0 must list active nodes in ascending order, each K times"

    # (2) test_pos sorted by (u,v)
    u_pos, v_pos = test_pos
    ok_order = ((u_pos[1:] - u_pos[:-1]) > 0) | \
                ((u_pos[1:] == u_pos[:-1]) & (v_pos[1:] >= v_pos[:-1]))
    assert bool(ok_order.all()), "test_pos is not sorted by (u,v)"

    # (3) deg consistency
    deg_from_pos = torch.bincount(u_pos, minlength=N)
    assert torch.equal(deg_from_pos, test_deg), "`deg` inconsistent with test_pos"
    
    print("All checks passed for test_pos and test_neg. Ogbl-citation2, safe!")


from typing import Dict, Literal
import torch
import numpy as np
from torch_geometric.datasets import ICEWS18

def load_icews18(root, type: Literal['min', 'mid', 'max']=None):
    # 加载训练、验证和测试数据
    train_dataset = ICEWS18(root, split='train')
    val_dataset = ICEWS18(root, split='val')
    test_dataset = ICEWS18(root, split='test')
    
    # 合并数据
    dataset = train_dataset
    dataset.data.sub = torch.cat([train_dataset.data.sub, val_dataset.data.sub, test_dataset.data.sub], dim=0)
    dataset.data.obj = torch.cat([train_dataset.data.obj, val_dataset.data.obj, test_dataset.data.obj], dim=0)
    dataset.data.rel = torch.cat([train_dataset.data.rel, val_dataset.data.rel, test_dataset.data.rel], dim=0)
    dataset.data.t = torch.cat([train_dataset.data.t, val_dataset.data.t, test_dataset.data.t], dim=0)
    
    # 初始化边集和时间存储
    edge_index = []
    edge_t = []
    time_dict = {}

    # 根据 type 选择时间处理方式
    if type == 'min':
        time_func = np.min
    elif type == 'mid':
        time_func = np.median
    elif type == 'max':
        time_func = np.max
    else:
        raise ValueError("Invalid type. Must be 'min', 'mid', or 'max'.")
    
    # 记录每个 (u, v) 边的所有时间
    for i, (u, v) in enumerate(zip(dataset.data.sub, dataset.data.obj)):
        pair = (u.item(), v.item())
        if pair not in time_dict:
            time_dict[pair] = []
        time_dict[pair].append(dataset.data.t[i].item())

    # 根据 `type` 计算每条边的时间值
    for pair in time_dict:
        times = np.array(time_dict[pair])
        selected_time = time_func(times)
        edge_index.append(pair)
        edge_t.append(selected_time)
    
    # 更新边集中的时间
    dataset.data.edge_index = torch.tensor(edge_index, dtype=torch.int64).t()
    dataset.data.edge_t = torch.tensor(edge_t, dtype=torch.float32)
    
    # 计算 split_time 字典
    split_time = {}
    sorted_times = np.sort(edge_t)
    total_edges = len(edge_t)
    
    for split_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        index = int(total_edges * split_ratio)
        split_time[split_ratio] = sorted_times[index - 1]  # 使用小于等于此时间的边占比为 split_ratio
    
    
    dataset.split_time = split_time
    
    return dataset

def load_csv_amazon(root):
    edges = pd.read_csv(root)
    print(f"表头: {edges.columns.tolist()}")
    edge_index = np.array(edges[['source', 'target']].values.T, dtype=np.int64) # shape (2, E)
    return edge_index

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

def transform_txt(train_data, test_data, node_feat=None):


    train_pos = [[], []]
    test_pos = []
    test_neg = []
    valid_pos = [[train_data[0][0].item(), train_data[0][0].item()], [train_data[0][1].item(), train_data[0][0].item()]] # 默认为空，不作数
    valid_neg = [[train_data[0][0].item(), train_data[0][0].item()], [train_data[0][1].item(), train_data[0][0].item()]] # 默认为空，不作数
    
    # train_data
    for u, v in zip(train_data[0], train_data[1]):
        train_pos[0].append(u)
        train_pos[1].append(v)
    
    # test_pos_data
    test_pos = test_data[0]
    test_neg = test_data[1]
    
    return train_pos, test_pos, test_neg, valid_pos, valid_neg


def write_txt(data, path):
    with open(path, 'w') as f:
        for u, v in zip(data[0], data[1]):
            f.write(f"{u} {v} {1}\n")

def split_dataset(name, split_ratio, data_type):
    K = 5
    
    node_feat = None
    if name == 'cora_ml':
        dataset = CitationFull(root="./datasets", name='cora_ml', to_undirected=False)
        split_subdataset_path = osp.join("./datasets", name, 'split')
    elif name == 'citeseer':
        dataset = CitationFull(root="./datasets", name='citeseer', to_undirected=False)
        split_subdataset_path = osp.join("./datasets", name, 'split')
    elif name == 'pubmed':
        dataset = CitationFull(root="./datasets", name='pubmed', to_undirected=False)
        split_subdataset_path = osp.join("./datasets", name, 'split')
    elif name == 'cora':
        dataset = CitationFull(root="./datasets", name='cora', to_undirected=False)
        split_subdataset_path = osp.join("./datasets", name, 'split')
    elif name == 'ogbl_citation2':
        test_type = 'edge'
        dataset = LinkPropPredDataset(root="./datasets", name='ogbl-citation2')
        node_feat = torch.from_numpy(dataset.graph['node_feat']).float()
        split_subdataset_path = osp.join("./datasets", name, 'split')
    elif name == 'ogbl_citation2_node':
        test_type = 'node'
        dataset = LinkPropPredDataset(root="./datasets", name='ogbl-citation2')
        node_feat = torch.from_numpy(dataset.graph['node_feat']).float()
        split_subdataset_path = osp.join("./datasets", 'ogbl_citation2', 'split_node')    
    elif name == 'twitter':
        dataset = np.load('datasets/twitter/edges.npy')
        node_feat = torch.from_numpy(np.load('datasets/twitter/features_top512.npy')).float()
        split_subdataset_path ='./datasets/twitter/split'
    elif 'icews18' in name:
        dataset = load_icews18(root = "./datasets/icews18", type=name.split('_')[-1])
        split_subdataset_path = osp.join("./datasets", name, 'split')
    elif name == 'amazon':
        edge_index = load_csv_amazon('./datasets/amazon/edges.csv')
        dataset = edge_index
        split_subdataset_path = './datasets/amazon/split'
    elif name == 'fly_larva':
        edge_index = load_csv_amazon('./datasets/fly_larva/edges.csv')
        dataset = edge_index
        split_subdataset_path = './datasets/fly_larva/split'
    elif name == 'yeast':
        edge_index = load_csv_amazon('./datasets/yeast/edges.csv')
        dataset = edge_index
        split_subdataset_path = './datasets/yeast/split'
    elif name == 'male':
        edge_index = load_csv_amazon('./datasets/male/edges.csv')
        dataset = edge_index
        split_subdataset_path = './datasets/male/split'
    
    if not osp.exists(split_subdataset_path):
        os.makedirs(split_subdataset_path)
    
    logging.info(f"Dataset: {name}, Split Subdataset Path: {split_subdataset_path}")
    if 'ogbl_citation2' in name:
        datawrapper = EmbeddingLinkPredictionDataWrapper_cit2(
            dataset, negative_ratio=K, 
            citation2=('ogbl_citation2' in name),
            test_type=test_type,
        )
    else:
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
    
    # 如果是 ogbl-citation2-node 数据集，就需要把 test 边排好序
    if name == 'ogbl_citation2_node' :
        # Convert to (E, 2) shape for sorting
        test_pos_edges = test_data[0].t()  # Shape: (E, 2)
        test_neg_edges = test_data[1].t()  # Shape: (E, 2)
        
        # Sort by first column (u) as primary key, second column (v) as secondary key
        # Use argsort with combined keys since lexsort is not available
        test_pos_combined_keys = test_pos_edges[:, 0] * (torch.max(test_pos_edges[:, 1]) + 1) + test_pos_edges[:, 1]
        test_pos_sorted_indices = torch.argsort(test_pos_combined_keys)
        
        test_neg_combined_keys = test_neg_edges[:, 0] * (torch.max(test_neg_edges[:, 1]) + 1) + test_neg_edges[:, 1]
        test_neg_sorted_indices = torch.argsort(test_neg_combined_keys)
        
        # Apply sorting and convert back to (2, E) tensor format
        test_pos_sorted = test_pos_edges[test_pos_sorted_indices].t()
        test_neg_sorted = test_neg_edges[test_neg_sorted_indices].t()
        
        test_data = (test_pos_sorted, test_neg_sorted)
        check_citation2_data(test_data[0], test_data[1], test_deg, N, K)  # 检查 test_pos 和 test_neg 的正确性
    

    
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
    elif data_type == 'txt':
        train_pos, test_pos, test_neg, valid_pos, valid_neg = transform_txt(train_data, test_data, node_feat=node_feat)
        split_subdataset_path = osp.join(split_subdataset_path, f'split_{split_ratio}')
        if not osp.exists(split_subdataset_path):
            os.makedirs(split_subdataset_path)
        write_txt(train_pos, osp.join(split_subdataset_path, 'train_pos.txt'))
        write_txt(test_pos, osp.join(split_subdataset_path, 'test_pos.txt'))
        write_txt(test_neg, osp.join(split_subdataset_path, 'test_neg.txt'))
        write_txt(valid_pos, osp.join(split_subdataset_path, 'valid_pos.txt'))
        write_txt(valid_neg, osp.join(split_subdataset_path, 'valid_neg.txt'))
        logging.info(f"Train Edges: {len(train_pos)}, Test Pos Edges: {len(test_pos)}, Test Neg Edges: {len(test_neg)}")
        logging.info(f"Train/Test Pos Edges: {len(train_pos) / (len(test_pos))}")
        logging.info(f"Train Edges/Nodes: {len(train_pos)} / {len(torch.unique(train_data.flatten()))} = {len(train_pos) / len(torch.unique(train_data.flatten()))}\n") # type: ignore
    
    print("Everything is done, check the logs for details.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split Dataset')
    parser.add_argument('--split_ratio', type=float, default=0.02)
    parser.add_argument('--dataset', type=str, default='ogbl-citation2',)
    parser.add_argument('--type', type=str, default='pt', choices=['pt', 'txt'])
    
    args = parser.parse_args()
    split_ratio = args.split_ratio
    dataset = args.dataset
    data_type = args.type
    
    logging.basicConfig(filename=f'./datasets/logs/split_{dataset}_ratio_{split_ratio}.log',
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    split_dataset(dataset, split_ratio, data_type)
    
    
    
