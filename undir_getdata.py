import csv
import logging
import os
import os.path as osp
import random
import sys

import cogdl
import numpy as np
import pandas as pd
import torch
from cogdl.datasets.planetoid_data import (CiteSeerDataset, CoraDataset,
                                           PubMedDataset)
from cogdl.datasets.ogb import (OGBLCollabDataset, OGBLPpaDataset)
from cogdl.wrappers.data_wrapper.link_prediction.embedding_link_prediction_dw import \
    EmbeddingLinkPredictionDataWrapper
from torch_geometric.datasets import ICEWS18
from typing import Literal
    
    
import warnings


warnings.filterwarnings("ignore", category=UserWarning)

import argparse


class Dataset:
    def __init__(self, root):
        print(f"Loading data from {root}")
        edge_index = self.load_csv(osp.join(root, 'edges.csv'))
        self.data = edge_index
        print("Data loaded")
    def load_csv(self, root):
        edges = pd.read_csv(root)
        print(f": {edges.columns.tolist()}")
        edge_index = np.array(edges[['source', 'target']].values.T, dtype=np.int64) # shape (2, E)
        return edge_index


def load_icews18(root, type: Literal['min', 'mid', 'max']=None):
    
    train_dataset = ICEWS18(root, split='train')
    val_dataset = ICEWS18(root, split='val')
    test_dataset = ICEWS18(root, split='test')
    
    
    dataset = train_dataset
    dataset.data.sub = torch.cat([train_dataset.data.sub, val_dataset.data.sub, test_dataset.data.sub], dim=0)
    dataset.data.obj = torch.cat([train_dataset.data.obj, val_dataset.data.obj, test_dataset.data.obj], dim=0)
    dataset.data.rel = torch.cat([train_dataset.data.rel, val_dataset.data.rel, test_dataset.data.rel], dim=0)
    dataset.data.t = torch.cat([train_dataset.data.t, val_dataset.data.t, test_dataset.data.t], dim=0)
    
    
    edge_index = []
    edge_t = []
    time_dict = {}

    
    if type == 'min':
        time_func = np.min
    elif type == 'mid':
        time_func = np.median
    elif type == 'max':
        time_func = np.max
    else:
        raise ValueError("Invalid type. Must be 'min', 'mid', or 'max'.")
    
    
    for i, (u, v) in enumerate(zip(dataset.data.sub, dataset.data.obj)):
        pair = (u.item(), v.item())
        if pair not in time_dict:
            time_dict[pair] = []
        time_dict[pair].append(dataset.data.t[i].item())

    
    for pair in time_dict:
        times = np.array(time_dict[pair])
        selected_time = time_func(times)
        edge_index.append(pair)
        edge_t.append(selected_time)
    
    
    dataset.data.edge_index = torch.tensor(edge_index, dtype=torch.int64).t()
    
    print("edge_index shape:", dataset.data.edge_index.shape)
    print("edge_t shape:", torch.tensor(edge_t, dtype=torch.float32).shape)
    dataset.data.edge_t = torch.tensor(edge_t, dtype=torch.float32)
    
    
    split_time = {}
    sorted_times = np.sort(edge_t)
    total_edges = len(edge_t)
    
    for split_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        index = int(total_edges * split_ratio)
        split_time[split_ratio] = sorted_times[index - 1]  
    
    
    dataset.split_time = split_time
    
    return dataset


def transform(train_data, test_data):
    train_pos = []
    test_pos = []
    test_neg = []
    valid_pos = [(train_data.edge_index[0][0].item(), train_data.edge_index[0][1].item())] 
    valid_neg = [(train_data.edge_index[0][0].item(), train_data.edge_index[0][1].item())] 
    
    # train_data
    for u, v in zip(train_data.edge_index[0], train_data.edge_index[1]):
        train_pos.append((u.item(), v.item()))
    
    # test_pos_data
    test_pos = test_data[0]
    test_neg = test_data[1]
    
    return train_pos, test_pos, test_neg, valid_pos, valid_neg

def write_txt(data, path, method):
    if method == "normal":
        pd.DataFrame(data).to_csv(path, header=None, index=None, sep='\t')
    elif method == "node2ket":
        with open(path, 'w') as f:
            for edge in data:
                f.write(f"{edge[0]} {edge[1]} {1}\n")

def split_dataset(name, method):
    if name == 'cora':
        dataset = CoraDataset("./undir_datasets/")
    elif name == 'citeseer':
        dataset = CiteSeerDataset("./undir_datasets/")
    elif name == 'pubmed':
        dataset = PubMedDataset("./undir_datasets/")    
    elif name == 'ogbl_collab':
        dataset = OGBLCollabDataset("./undir_datasets/")
    elif name == 'ogbl_ppa':
        dataset = OGBLPpaDataset("./undir_datasets/")
    elif 'icews18' in name:
        dataset = load_icews18(root = "./undir_datasets/icews18", type=name.split('_')[-1])
    else:
        dataset = Dataset(f"./undir_datasets/{name}")
    
    datawrapper = EmbeddingLinkPredictionDataWrapper(dataset, negative_ratio=5, icews18=('icews18' in name))
    split_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.02]
    if dataset == 'ogbl_collab':
        split_ratios += [0.99, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
    
      
    for split_ratio in split_ratios:
        logging.info(f"Splitting ratio: {split_ratio}")

        # >>> get the data processed by CogDL >>> #
        if method == 'normal':
            split_subdataset_path = osp.join("./undir_datasets/", name, 'split_' + str(split_ratio))
        elif method == 'node2ket':
            split_subdataset_path = osp.join("./undir_datasets/", 'data', name, 
                                             'split_' + str(split_ratio))
        
        datawrapper.pre_transform(split_ratio)
        train_data = datawrapper.train_wrapper()
        test_data = datawrapper.test_wrapper()
        
        # >>> get the data >>> #
        train_pos, test_pos, test_neg, valid_pos, valid_neg = transform(train_data, test_data)
        
        # >>> save the data >>> #
        if not osp.exists(split_subdataset_path):
            os.makedirs(split_subdataset_path)
        
        write_txt(train_pos, osp.join(split_subdataset_path, 'train_pos.txt'), method)
        write_txt(test_pos, osp.join(split_subdataset_path, 'test_pos.txt'), method)
        write_txt(test_neg, osp.join(split_subdataset_path, 'test_neg.txt'), method)
        write_txt(valid_pos, osp.join(split_subdataset_path, 'valid_pos.txt'), method)
        write_txt(valid_neg, osp.join(split_subdataset_path, 'valid_neg.txt'), method)
        
        logging.info(f"Train Edges: {len(train_pos)}, Test Pos Edges: {len(test_pos)}, Test Neg Edges: {len(test_neg)}")
        logging.info(f"Train/Test Pos Edges: {len(train_pos) / (len(test_pos))}")
        logging.info(f"Train Edges/Nodes: {len(train_pos)} / {len(torch.unique(torch.cat(train_data.edge_index)))} = {len(train_pos) / len(torch.unique(torch.cat(train_data.edge_index)))}\n")
        
# utils 
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--method', type=str, default="normal", choices=["normal", "node2ket"])
    
    args = parser.parse_args()
    name = args.dataset
    method = args.method
    
    
    
    logging.basicConfig(filename=f'dataset_split_n{name}_m{method}.log', level=logging.INFO, filemode='w')
    set_random_seed(0)
    # dataset_names = ['cora', 'citeseer', 'pubmed', 'ogbl_ppa', 'ogbl_collab']

    logging.info(f"\n>>>>> {name} is splitting >>>>>\n")
    split_dataset(name, method)
    logging.info(f"\n<<<<< {name} splitting is Finished <<<<<\n")
    
    
    
