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
from torch_geometric.datasets import ICEWS18
from torch_geometric.datasets import CitationFull
import argparse

np.random.seed(0)
random.seed(0)

def load_icews18(root):
    train_dataset = ICEWS18(root, split='train')
    val_dataset = ICEWS18(root, split='val')
    test_dataset = ICEWS18(root, split='test')
    
    dataset = train_dataset
    dataset.data.sub = torch.cat([train_dataset.data.sub, val_dataset.data.sub, test_dataset.data.sub], dim=0)
    dataset.data.obj = torch.cat([train_dataset.data.obj, val_dataset.data.obj, test_dataset.data.obj], dim=0)
    dataset.data.rel = torch.cat([train_dataset.data.rel, val_dataset.data.rel, test_dataset.data.rel], dim=0)
    dataset.data.t = torch.cat([train_dataset.data.t, val_dataset.data.t, test_dataset.data.t], dim=0)
        
    edge_index = torch.tensor(torch.zeros((dataset.data.sub.shape[0], 2)), dtype=torch.int64)
    for i, (u, v) in enumerate(zip(dataset.data.sub, dataset.data.obj)):
        edge_index[i][0] = u
        edge_index[i][1] = v
    dataset.data.edge_index = edge_index.t()
    return dataset

def transform_pt(train_data, test_data):
    train = {'edge': []}
    test = {'edge': [], 'edge_neg': []}
    valid = {'edge': [], 'edge_neg': []}

    print("processing train data")
    # train data
    for u, v in zip(train_data.edge_index[0], train_data.edge_index[1]):
        train['edge'].append([u.item(), v.item()])
        
    print("processing test data")
    # test data
    test['edge'] = test_data[0]
    test['edge_neg'] = test_data[1]
    
    print("processing valid data")
    # valid data
    valid['edge'] = [[train['edge'][0][0], train['edge'][0][1]]]
    valid['edge_neg'] = [[train['edge'][0][0], train['edge'][0][1]]]
    
    print("processing complete")
    
    train['edge'] = torch.tensor(train['edge']) # type: ignore
    test['edge'] = torch.tensor(test['edge']) # type: ignore
    test['edge_neg'] = torch.tensor(test['edge_neg']) # type: ignore
    valid['edge'] = torch.tensor(valid['edge']) # type: ignore
    valid['edge_neg'] = torch.tensor(valid['edge_neg']) # type: ignore
    
    logging.info(f"""Train Edges: {train['edge'].shape[0]}, 
                 Test Pos Edges: {test['edge'].shape[0]}, Test Neg Edges: {test['edge_neg'].shape[0]}""")
    
    return train, test, valid

def transform_txt(train_data, test_data):
    train_pos = []
    test_pos = []
    test_neg = []
    valid_pos = [(train_data.edge_index[0][0].item(), train_data.edge_index[0][1].item())] # 默认为空，不作数
    valid_neg = [(train_data.edge_index[0][0].item(), train_data.edge_index[0][1].item())] # 默认为空，不作数
    
    # train_data
    for u, v in zip(train_data.edge_index[0], train_data.edge_index[1]):
        train_pos.append((u.item(), v.item()))
    
    # test_pos_data
    test_pos = test_data[0]
    test_neg = test_data[1]
    
    return train_pos, test_pos, test_neg, valid_pos, valid_neg


def write_txt(data, path):
    with open(path, 'w') as f:
        for edge in data:
            f.write(f"{edge[0]} {edge[1]} {1}\n")

def split_dataset(name, split_ratio, data_type):
    if name == 'cora_ml':
        dataset = CitationFull(root="./datasets", name='cora_ml', to_undirected=False)
        split_subdataset_path = osp.join("./datasets", name, 'split_ori')
    elif name == 'citeseer':
        dataset = CitationFull(root="./datasets", name='citeseer', to_undirected=False)
        split_subdataset_path = osp.join("./datasets", name, 'split_ori')
    elif name == 'pubmed':
        dataset = CitationFull(root="./datasets", name='pubmed', to_undirected=False)
        split_subdataset_path = osp.join("./datasets", name, 'split_ori')
    elif name == 'cora':
        dataset = CitationFull(root="./datasets", name='cora', to_undirected=False)
        split_subdataset_path = osp.join("./datasets", name, 'split_ori')
    
    # elif name == 'icews18':
    #     dataset = load_icews18(root = "./datasets/icews18")
    #     split_subdataset_path = osp.join("./datasets", name, 'split')
    
    if not osp.exists(split_subdataset_path):
        os.makedirs(split_subdataset_path)
    
    logging.info(f"Dataset: {name}, Split Subdataset Path: {split_subdataset_path}")
    datawrapper = EmbeddingLinkPredictionDataWrapper(dataset, negative_ratio=5, icews18=(name == 'icews18'))
    logging.info(f"Splitting ratio: {split_ratio}")

    # >>> Get the train/test split >>> #
    datawrapper.pre_transform(split_ratio)
    train_data = datawrapper.train_wrapper()
    test_data = datawrapper.test_wrapper()
    
    # >>> Transform and Save >>> #
    if data_type == 'pt':
        train, test, valid = transform_pt(train_data, test_data)
        torch.save({'train': train, 'test': test, 'valid': valid}, 
            osp.join(split_subdataset_path, f'split_dict_{split_ratio}.pt'))
        logging.info(f"""Train Edges: {train['edge'].shape[0]}, Test Pos Edges: {test['edge'].shape[0]}, Test Neg Edges: {test['edge_neg'].shape[0]}""") # type: ignore
        logging.info(f"""Train/Test Pos Edges: {train['edge'].shape[0] / test['edge'].shape[0]}""") # type: ignore
        logging.info(f"""Train Edges/Nodes: {len(train['edge'])} / {len(torch.unique(train['edge']))} 
                    = {len(train['edge']) / len(torch.unique(torch.unique(train['edge'])))}\n""")
    elif data_type == 'txt':
        train_pos, test_pos, test_neg, valid_pos, valid_neg = transform_txt(train_data, test_data)
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
        logging.info(f"Train Edges/Nodes: {len(train_pos)} / {len(torch.unique(torch.cat(train_data.edge_index)))} = {len(train_pos) / len(torch.unique(torch.cat(train_data.edge_index)))}\n") # type: ignore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split Dataset')
    parser.add_argument('--split_ratio', type=float, default=0.02)
    parser.add_argument('--dataset', type=str, required=True)
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
    
    
    
