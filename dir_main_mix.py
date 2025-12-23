import numpy as np
import os
import torch
import warnings
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import *
import pandas as pd
import networkx as nx
import random
warnings.filterwarnings('ignore', 'divide by zero encountered in log2')
import torch
import argparse
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# >>> Dataset and Model <<< #
class MyDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        sta_ind = self.indices[idx]
        return sta_ind.clone().detach().to(dtype=torch.int64).to(device)
class CritiGraph(torch.nn.Module):
    def __init__(self, h, tp, c, eps, neg, gamma, r, ty, alpha, epoch, batch_size, pos_ratio, 
                 chunks, convergence, eval_step):
        super().__init__() 
        self.h = h
        self.tp = tp
        self.n = int(2**h)
        self.c = c
        self.r = r
        self.ty = ty
        self.k = int(c*h)
        self.eps = eps
        self.neg = neg
        self.gamma = gamma
        self.alpha = alpha
        self.epoch = epoch  
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.flip_masks = (1 << torch.arange(self.h, dtype=torch.int64, device=device)).unsqueeze(0).unsqueeze(2)
        self.distance_lookup_table = self.generate_distance_lookup_table()
        self.chunks = chunks
        self.convergence = convergence
        self.eval_step = eval_step
        if self.ty[:6] == 'single':
            self.rge, self.dvd, self.nu = 4, 2, 3
        elif self.ty[:6] == 'double':
            self.rge, self.dvd, self.nu = 6, 3, 5
    def generate_distance_lookup_table(self):
        xor_results = torch.arange(self.n, dtype=torch.int64, device=device)
        return torch.where(xor_results == 0, 
                           torch.tensor(0, dtype=torch.int64, device=device), 
                           torch.floor(torch.log2(xor_results.float())) + 1).int()
    def distance(self, coord1, coord2):
        xor_result = torch.bitwise_xor(coord1, coord2)
        return self.distance_lookup_table[xor_result]
    def generate_random_masks(self, sz):
        upper_bounds = 2**torch.arange(self.h, dtype=torch.int64, device=device)
        random_numbers = torch.randint(0, self.n, (self.h, sz, self.k, self.tp), dtype=torch.int64, device=device)
        masks = random_numbers % upper_bounds.view(-1, 1, 1, 1)
        return masks.permute(1, 0, 2, 3)
    def connection(self, ori_int):
        flipped_ints = ori_int.unsqueeze(1) ^ self.flip_masks
        random_masks = self.generate_random_masks(flipped_ints.size(0))
        result = (flipped_ints.unsqueeze(2) ^ random_masks).view(flipped_ints.size(0), self.h*self.k, self.tp)
        return torch.cat((result, ori_int.unsqueeze(1)), dim=1) 
    def heaviside(self, x):
        return torch.where(x <= 0, torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
    def P(self, dis, ig1, ig2, key):
        p_12 = self.p(dis, ig1, ig2)
        p_21 = self.p(dis, ig2, ig1)
        if key == 0:
            return p_12
        elif key == 1:
            return p_21
        elif key == 2:
            return 1-p_12
            # return (1-p_12)*(1-p_21)
        elif key == 3:
            return 1-p_21
            # return (1-p_21)*(1-p_12)
    def p(self, dis, ig1, ig2):       
        deg1, deg2 = self.out_degree[ig1], self.in_degree[ig2]
        ap = (deg1+1)*(deg2+1)
        aa = (dis+self.eps)/torch.log(ap)[:,:,None,None]
        return 1/(1+aa**self.gamma/self.alpha)
    def p_test(self, dis, ig1, ig2):
        deg1, deg2 = self.out_degree[ig1], self.in_degree[ig2]
        ap = (deg1+1)*(deg2+1)
        aa = (dis+self.eps)/torch.log(ap)
        return 1/(1+aa**self.gamma/self.alpha)
    def cs(self, num):
        if self.ty[:6] == 'single':
            if num < 3:
                return 0
            else:
                return 1
        elif self.ty[:6] == 'double':
            if num in {0,1,3}:
                return 0
            else:
                return 1
    def ce(self, num):
        if self.ty[:6] == 'single':
            if num != 1:
                return 0
            else:
                return 1
        elif self.ty[:6] == 'double':
            if num in {0,3,4}:
                return 0
            else:
                return 1
    def cha(self, num):
        if num < 1:
            return 0
        else:
            return 1
    def loom(self, epoch, sta_ind):
        pos_ind, lg, mask = self.neighbor_batch(sta_ind, epoch)
        sta_loc = [self.locations[self.cs(j)][sta_ind] for j in range(self.rge)]
        with torch.no_grad():
            pos_loc = [torch.full((pos_ind[j//self.dvd].size(0), pos_ind[j//self.dvd].size(1), self.tp), -1, dtype=torch.int64, device=device) for j in range(self.rge)]
            for j in range(self.rge):
                pos_loc[j][mask[j//self.dvd]] = self.locations[self.ce(j)][pos_ind[j//self.dvd][mask[j//self.dvd]]]
            neg_ind = [torch.full(pos_ind[j].size(), -1, dtype=torch.int64, device=device) for j in range(2)]
            for j in range(2):
                neg_ind[j][mask[j]] = torch.randint(0, self.num_nodes, (pos_ind[j].size(0), pos_ind[j].size(1)), dtype=torch.int64, device=device)[mask[j]]
            neg_loc = [torch.full((pos_ind[j//self.dvd].size(0), pos_ind[j//self.dvd].size(1), self.tp), -1, dtype=torch.int64, device=device) for j in range(self.rge)]
            for j in range(self.rge):
                neg_loc[j][mask[j//self.dvd]] = self.locations[self.ce(j)][neg_ind[j//self.dvd][mask[j//self.dvd]]]
            
            cnc_loc_2 = [self.connection(self.locations[j][sta_ind]) for j in range(2)]
            indices = [torch.randperm(cnc_loc_2[j].size(1)) for j in range(2)]
            cnc_loc = [cnc_loc_2[self.cs(j)][:, indices[self.cs(j)], :] for j in range(self.rge)]
            
            mask1 = [mask[j//self.dvd].unsqueeze(2).repeat(1, 1, self.tp) for j in range(self.rge)]
            mask2 = [mask1[j].unsqueeze(2).repeat(1, 1, cnc_loc[j].size(1),1) for j in range(self.rge)]
            dis_sta_pos = [torch.where(mask1[j], self.distance(sta_loc[j][:,None,:], pos_loc[j]), -1) for j in range(self.rge)]
            dis_sta_posum = [torch.sum(dis_sta_pos[j], dim=-1) for j in range(self.rge)]
            dis_sta_neg = [torch.where(mask1[j], self.distance(sta_loc[j][:,None,:], neg_loc[j]), -1) for j in range(self.rge)]
            dis_sta_negum = [torch.sum(dis_sta_neg[j], dim=-1) for j in range(self.rge)]
            dis_pos_cnc = [torch.where(mask2[j], self.distance(cnc_loc[j][:,None,:,:], pos_loc[j][:,:,None,:]), -1) for j in range(self.rge)]
            dis_neg_cnc = [torch.where(mask2[j], self.distance(cnc_loc[j][:,None,:,:], neg_loc[j][:,:,None,:]), -1) for j in range(self.rge)]
            dis_new_pos_0 = [torch.where(mask2[j], (dis_pos_cnc[j]-dis_sta_pos[j][:,:,None,:]+dis_sta_posum[j][:,:,None,None])/self.tp, 0) for j in range(self.rge)]
            dis_new_neg_0 = [torch.where(mask2[j], (dis_neg_cnc[j]-dis_sta_neg[j][:,:,None,:]+dis_sta_negum[j][:,:,None,None])/self.tp, 1000) for j in range(self.rge)]
            if self.ty[-1] == 'd':
                if self.ty[:6] == 'single':
                    dis_new_pos = [self.r*dis_new_pos_0[0]+(1-self.r)*dis_new_pos_0[1],
                                    self.r*dis_new_pos_0[2]+(1-self.r)*dis_sta_pos[3][:,:,None,:],
                                    self.r*dis_sta_pos[2][:,:,None,:]+(1-self.r)*dis_new_pos_0[3]]
                    dis_new_neg = [self.r*dis_new_neg_0[0]+(1-self.r)*dis_new_neg_0[1],
                                    self.r*dis_new_neg_0[2]+(1-self.r)*dis_sta_neg[3][:,:,None,:],
                                    self.r*dis_sta_neg[2][:,:,None,:]+(1-self.r)*dis_new_neg_0[3]]
                    pos_loss_0 = [-torch.sum(torch.where(mask2[j+1], torch.log(self.eps+self.P(dis_new_pos[j], sta_ind[:,None], pos_ind[self.cha(j)], self.cha(j))), torch.tensor(0.0, device=device)), dim=1) / lg[self.cha(j)][:,None,None] for j in range(3)]
                    neg_loss_0 = [-torch.sum(torch.where(mask2[j+1], torch.log(self.eps+self.P(dis_new_neg[j], sta_ind[:,None], neg_ind[self.cha(j)], self.cha(j)+2)), torch.tensor(0.0, device=device)), dim=1) / lg[self.cha(j)][:,None,None] for j in range(3)]
                    pos_loss = [pos_loss_0[0] + pos_loss_0[1], pos_loss_0[2]]
                    neg_loss = [neg_loss_0[0] + neg_loss_0[1], neg_loss_0[2]]
                    
                elif self.ty[:6] == 'double':
                    dis_new_pos = [self.r*(dis_new_pos_0[0]+dis_sta_pos[2][:,:,None,:])/2+(1-self.r)*dis_new_pos_0[1], 
                                    self.r*(dis_sta_pos[0][:,:,None,:]+dis_new_pos_0[2])/2+(1-self.r)*dis_sta_pos[1][:,:,None,:],
                                    self.r*(dis_new_pos_0[3]+dis_sta_pos[4][:,:,None,:])/2+(1-self.r)*dis_sta_pos[5][:,:,None,:],
                                    self.r*(dis_sta_pos[3][:,:,None,:]+dis_new_pos_0[4])/2+(1-self.r)*dis_new_pos_0[5]]
                    dis_new_neg = [self.r*(dis_new_neg_0[0]+dis_sta_neg[2][:,:,None,:])/2+(1-self.r)*dis_new_neg_0[1],
                                    self.r*(dis_sta_neg[0][:,:,None,:]+dis_new_neg_0[2])/2+(1-self.r)*dis_sta_neg[1][:,:,None,:],
                                    self.r*(dis_new_neg_0[3]+dis_sta_neg[4][:,:,None,:])/2+(1-self.r)*dis_sta_neg[5][:,:,None,:],
                                    self.r*(dis_sta_neg[3][:,:,None,:]+dis_new_neg_0[4])/2+(1-self.r)*dis_new_neg_0[5]]
                    pos_loss_0 = [-torch.sum(torch.where(mask2[j+1], torch.log(self.eps+self.P(dis_new_pos[j], sta_ind[:,None], pos_ind[j//2], j//2)), torch.tensor(0.0, device=device)), dim=1) / lg[j//2][:,None,None] for j in range(4)]
                    neg_loss_0 = [-torch.sum(torch.where(mask2[j+1], torch.log(self.eps+self.P(dis_new_neg[j], sta_ind[:,None], neg_ind[j//2], j//2+2)), torch.tensor(0.0, device=device)), dim=1) / lg[j//2][:,None,None] for j in range(4)]
                    pos_loss = [pos_loss_0[0]+pos_loss_0[2], pos_loss_0[1]+pos_loss_0[3]]
                    neg_loss = [neg_loss_0[0]+neg_loss_0[2], neg_loss_0[1]+neg_loss_0[3]]
                    
            else:
                dis_new_pos, dis_new_neg = dis_new_pos_0, dis_new_neg_0
                p_pos = [torch.where(mask2[j], self.P(dis_new_pos[j], sta_ind[:,None], pos_ind[j//self.dvd], j//self.dvd), torch.tensor(0.0, device=device)) for j in range(self.rge)]
                p_neg = [torch.where(mask2[j], self.P(dis_new_neg[j], sta_ind[:,None], neg_ind[j//self.dvd], j//self.dvd+2), torch.tensor(0.0, device=device)) for j in range(self.rge)]
                if self.ty[-1] == 'p':
                    if self.ty[:6] == 'single':
                        p_pos_1 = [torch.where(mask2[j+2], self.P(dis_sta_pos[j+2][:,:,None,:], sta_ind[:,None], pos_ind[j//2+1], j//2+1), torch.tensor(0.0, device=device)) for j in range(2)]
                        p_neg_1 = [torch.where(mask2[j+2], self.P(dis_sta_neg[j+2][:,:,None,:], sta_ind[:,None], neg_ind[j//2+1], j//2+3), torch.tensor(0.0, device=device)) for j in range(2)]
                        
                        pos_loss = [-torch.sum(torch.where(mask2[0], torch.log(self.eps+self.r*p_pos[0]+(1-self.r)*p_pos[1]), torch.tensor(0.0, device=device)), dim=1) / lg[0][:,None,None]\
                                    -torch.sum(torch.where(mask2[2], torch.log(self.eps+self.r*p_pos[2]+(1-self.r)*p_pos_1[1]), torch.tensor(0.0, device=device)), dim=1) / lg[1][:,None,None],
                                    -torch.sum(torch.where(mask2[3], torch.log(self.eps+self.r*p_pos_1[0]+(1-self.r)*p_pos[3]), torch.tensor(0.0, device=device)), dim=1) / lg[1][:,None,None]]
                        neg_loss = [-torch.sum(torch.where(mask2[0], torch.log(self.eps+self.r*p_neg[0]+(1-self.r)*p_neg[1]), torch.tensor(0.0, device=device)), dim=1) / lg[0][:,None,None]\
                                    -torch.sum(torch.where(mask2[2], torch.log(self.eps+self.r*p_neg[2]+(1-self.r)*p_neg_1[1]), torch.tensor(0.0, device=device)), dim=1) / lg[1][:,None,None],
                                    -torch.sum(torch.where(mask2[3], torch.log(self.eps+self.r*p_neg_1[0]+(1-self.r)*p_neg[3]), torch.tensor(0.0, device=device)), dim=1) / lg[1][:,None,None]]
                    elif self.ty[:6] == 'double':
                        dis_new_pos = [self.r*(dis_new_pos_0[0]+dis_sta_pos[2][:,:,None,:])/2+(1-self.r)*dis_new_pos_0[1], 
                                    self.r*(dis_sta_pos[0][:,:,None,:]+dis_new_pos_0[2])/2+(1-self.r)*dis_sta_pos[1][:,:,None,:],
                                    self.r*(dis_new_pos_0[3]+dis_sta_pos[4][:,:,None,:])/2+(1-self.r)*dis_sta_pos[5][:,:,None,:],
                                    self.r*(dis_sta_pos[3][:,:,None,:]+dis_new_pos_0[4])/2+(1-self.r)*dis_new_pos_0[5]]
                        p_pos_1 = [torch.where(mask2[j], self.P(dis_sta_pos[j][:,:,None,:], sta_ind[:,None], pos_ind[j//3], j//3), torch.tensor(0.0, device=device)) for j in range(6)]
                        p_neg_1 = [torch.where(mask2[j], self.P(dis_sta_neg[j][:,:,None,:], sta_ind[:,None], neg_ind[j//3], j//3+2), torch.tensor(0.0, device=device)) for j in range(6)]
                        #此后要改
                        pos_loss = [-torch.sum(torch.where(mask2[0], torch.log(self.eps+self.r*p_pos[0]+(1-self.r)*p_pos[1]), torch.tensor(0.0, device=device)), dim=1) / lg[0][:,None,None]\
                                    -torch.sum(torch.where(mask2[3], torch.log(self.eps+p_pos[3]), torch.tensor(0.0, device=device)), dim=1) / lg[1][:,None,None],
                                    -torch.sum(torch.where(mask2[5], torch.log(self.eps+self.r*p_pos[5]+(1-self.r)*p_pos[4]), torch.tensor(0.0, device=device)), dim=1) / lg[1][:,None,None]]\
                                    -torch.sum(torch.where(mask2[2], torch.log(self.eps+p_pos[2]), torch.tensor(0.0, device=device)), dim=1) / lg[0][:,None,None]
                        neg_loss = [-torch.sum(torch.where(mask2[0], torch.log(self.eps+self.r*p_neg[0]+(1-self.r)*p_neg[1]), torch.tensor(0.0, device=device)), dim=1) / lg[0][:,None,None]\
                                    -torch.sum(torch.where(mask2[3], torch.log(self.eps+p_neg[3]), torch.tensor(0.0, device=device)), dim=1) / lg[1][:,None,None],
                                    -torch.sum(torch.where(mask2[5], torch.log(self.eps+self.r*p_neg[5]+(1-self.r)*p_pos[4]), torch.tensor(0.0, device=device)), dim=1) / lg[1][:,None,None]]\
                                    -torch.sum(torch.where(mask2[2], torch.log(self.eps+p_neg[2]), torch.tensor(0.0, device=device)), dim=1) / lg[0][:,None,None]    
                elif self.ty[-1] == 'l':
                    pos_loss_0 = [-torch.sum(torch.where(mask2[j], torch.log(self.eps+p_pos[j]), torch.tensor(0.0, device=device)), dim=1) / lg[j//self.dvd][:,None,None] for j in range(self.rge)]
                    neg_loss_0 = [-torch.sum(torch.where(mask2[j], torch.log(self.eps+p_neg[j]), torch.tensor(0.0, device=device)), dim=1) / lg[j//self.dvd][:,None,None] for j in range(self.rge)]
                    if self.ty[:6] == 'single':
                        pos_loss = [self.r*pos_loss_0[0]+(1-self.r)*pos_loss_0[1]+self.r*pos_loss_0[2],(1-self.r)*pos_loss_0[3]]
                        neg_loss = [self.r*neg_loss_0[0]+(1-self.r)*neg_loss_0[1]+self.r*neg_loss_0[2],(1-self.r)*neg_loss_0[3]]
                    elif self.ty[:6] == 'double':
                        pos_loss = [self.r*(pos_loss_0[0]+pos_loss_0[3])/2+(1-self.r)*pos_loss_0[1], self.r*(pos_loss_0[2]+pos_loss_0[5])/2+(1-self.r)*pos_loss_0[4]]
                        neg_loss = [self.r*(neg_loss_0[0]+neg_loss_0[3])/2+(1-self.r)*neg_loss_0[1], self.r*(neg_loss_0[2]+neg_loss_0[5])/2+(1-self.r)*neg_loss_0[4]]
           
            total_loss = self.pos_ratio * pos_loss + neg_loss
            index = [torch.argmin(total_loss[j], dim=1) for j in range(2)]
            i_indices_head, j_indices_head = torch.meshgrid(torch.arange(sta_ind.size(0)), torch.arange(self.tp), indexing='ij')
            i_indices_tail, j_indices_tail = torch.meshgrid(torch.arange(sta_ind.size(0)), torch.arange(self.tp), indexing='ij')
            self.locations[0][sta_ind[i_indices_head], j_indices_head] = cnc_loc[0][i_indices_head, index[0][i_indices_head, j_indices_head], j_indices_head]
            self.locations[1][sta_ind[i_indices_tail], j_indices_tail] = cnc_loc[self.nu][i_indices_tail, index[1][i_indices_tail, j_indices_tail], j_indices_tail]
            tl = torch.mean(total_loss[0][i_indices_head, index[0][i_indices_head, j_indices_head], j_indices_head]) + torch.mean(total_loss[1][i_indices_tail, index[1][i_indices_tail, j_indices_tail], j_indices_tail])
            pl = torch.mean(pos_loss[0][i_indices_head, index[0][i_indices_head, j_indices_head], j_indices_head]) + torch.mean(pos_loss[1][i_indices_tail, index[1][i_indices_tail, j_indices_tail], j_indices_tail])
            nl = torch.mean(neg_loss[0][i_indices_head, index[0][i_indices_head, j_indices_head], j_indices_head]) + torch.mean(neg_loss[1][i_indices_tail, index[1][i_indices_tail, j_indices_tail], j_indices_tail])
        
        return tl, pl, nl

    def get_neighbor(self):
        neighbor_out = [torch.tensor(list(set(self.G.successors(ii))), dtype=torch.int64, device=device)
            for ii in range(self.num_nodes)]
        neighbor_in = [torch.tensor(list(set(self.G.predecessors(ii))), dtype=torch.int64, device=device)
            for ii in range(self.num_nodes)]
        neighbor = [neighbor_out, neighbor_in]
        neighbor_dict = [{ii: neighbor[j][ii] for ii in range(self.num_nodes)} for j in range(2)]
        neighbor_tensor = [torch.full((self.num_nodes, self.max_degree), -1, dtype=torch.int64, device=device) for _ in range(2)]
        for j in range(2):
            for n, nbs in neighbor_dict[j].items():
                neighbor_tensor[j][n, :len(nbs)] = nbs
        return neighbor_tensor
    def neighbor_batch(self, sta_ind, epoch):
        
        bs = sta_ind.size(0)
        batch_max_degree = self.degree[sta_ind].max().item()
        batch_neighbor = [self.neighbor_tensor[j][sta_ind, :batch_max_degree] for j in range(2)]
        batch_lengths = [self.out_degree[sta_ind], 
                        self.in_degree[sta_ind]]
        if epoch <= self.convergence * self.epoch:
            random_probs = torch.rand(bs, device=device) # (bs, )
            choosing_mask = random_probs > 0.2 # (bs, )
            batch_lengths = [torch.where(choosing_mask, batch_lengths[j], 1) for j in range(2)]
            
            one_random_neighbor = [(torch.rand(bs, device=device) * batch_lengths[j]).floor().long() for j in range(2)] # (bs, )
            random_neighbor_mask = [torch.zeros((bs, batch_max_degree), dtype=torch.bool, device=device) for _ in range(2)] # (bs, max_degree)
            for j in range(2):
                random_neighbor_mask[j][torch.arange(bs), one_random_neighbor[j]] = True # (bs, max_degree)
                one_random_neighbor[j] = torch.full((bs, batch_max_degree), -1, dtype=torch.int64, device=device)
                one_random_neighbor[j][:, 0] = batch_neighbor[j][random_neighbor_mask[j]] # (bs, max_degree)            
                batch_neighbor[j] = torch.where(choosing_mask.unsqueeze(1), 
                                            batch_neighbor[j],
                                            one_random_neighbor[j]) # (bs, max_degree)
        batch_mask = [batch_neighbor[j] != -1 for j in range(2)]
        for j in range(2):
            batch_lengths[j] = torch.where(batch_lengths[j] == 0, 
                                torch.tensor(1, dtype=torch.int64, device=device), 
                                batch_lengths[j])
        return batch_neighbor, batch_lengths, batch_mask
    
    def forward(self, graph, test_pos, test_neg):
        current_time = datetime.now()
        print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
        print("start to load data")
        self.Go = nx.DiGraph()
        for edge in graph:
            self.Go.add_edge(edge[0].item(), edge[1].item())
        old_nodes = list(self.Go.nodes())
        self.mapping = {old_node: new_node for new_node, old_node in enumerate(old_nodes)}
        num_old_nodes = max(old_nodes) + 10  # Ensure we accommodate all possible indices  
        self.bucket_tensor = torch.full((num_old_nodes,), -1, dtype=torch.long)  
        for new_node, old_node in enumerate(old_nodes):  
            self.bucket_tensor[old_node] = new_node 
        self.bucket_tensor = self.bucket_tensor.cuda()
        self.G = nx.relabel_nodes(self.Go, self.mapping)
        self.num_nodes = self.G.number_of_nodes()
        self.num_edges = self.G.number_of_edges()
        print('num_nodes', self.num_nodes)
        print('num_edges', self.num_edges)
        print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
        self.out_degree = torch.tensor([self.G.out_degree(n) for n in self.G.nodes()], dtype=torch.int64, device=device)
        self.in_degree = torch.tensor([self.G.in_degree(n) for n in self.G.nodes()], dtype=torch.int64, device=device)
        self.degree = torch.tensor([self.G.degree(n) for n in self.G.nodes()], dtype=torch.int64, device=device)
        self.max_degree = self.degree.max()
        self.neighbor_tensor = self.get_neighbor()
        self.locations_head = torch.randint(0, self.n, (self.num_nodes, self.tp), dtype=torch.int64, device=device)
        self.locations_tail = torch.randint(0, self.n, (self.num_nodes, self.tp), dtype=torch.int64, device=device)
        self.locations = [self.locations_head, self.locations_tail]
        self.li = torch.arange(self.num_nodes, dtype=torch.int64, device=device)[self.degree>0]
        dataset = MyDataset(self.li)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epoch):    
            current_time = datetime.now()
            print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "pos+neg, pos, neg")
            sys.stdout.flush()
            total, positive, negative, ite = 0, 0, 0, 0
            for batch in dataloader:
                tot, pos, neg = self.loom(epoch, batch)
                total += tot
                positive += pos
                negative += neg
                ite += 1
            total /= ite
            positive /= ite
            negative /= ite
            print(epoch, total.device, total.item(), positive.item(), negative.item())
            if epoch % self.eval_step == 0 or epoch == self.epoch - 1:
                print("current_time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print("start to eval")
                sys.stdout.flush()
                self.eval(test_pos, test_neg, epoch)
    
    def get_score(self, data):
        dt = data.t()
        dt1 = self.bucket_tensor[dt[0]]
        dt2 = self.bucket_tensor[dt[1]]
        N = dt1.shape[0]
        chunk_size = (N + self.chunks - 1) // self.chunks  # ceil division for 16 chunks
        pp_chunks = []
        for i in range(self.chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, N)
            if start >= end:
                break
            chunk_dt1 = dt1[start:end]
            chunk_dt2 = dt2[start:end]
            chunk_vector_head_out = self.locations[0][chunk_dt1]
            chunk_vector_head_in = self.locations[0][chunk_dt2]
            chunk_vector_tail_out = self.locations[1][chunk_dt1]
            chunk_vector_tail_in = self.locations[1][chunk_dt2]
            dis_hh = self.distance(chunk_vector_head_out, chunk_vector_head_in).float()
            dis_ht = self.distance(chunk_vector_head_out, chunk_vector_tail_in).float()
            dis_tt = self.distance(chunk_vector_tail_out, chunk_vector_tail_in).float()
            dis_hh_mean = torch.mean(dis_hh, dim=-1)
            dis_ht_mean = torch.mean(dis_ht, dim=-1)
            dis_tt_mean = torch.mean(dis_tt, dim=-1)
            if self.ty[-1] == 'd':
                if self.ty[:6] == 'single':
                    dis_mean = self.r * dis_hh_mean + (1 - self.r) * dis_ht_mean
                elif self.ty[:6] == 'double':
                    dis_mean = self.r * (dis_hh_mean + dis_tt_mean) / 2 + (1 - self.r) * dis_ht_mean
                pp_chunk = self.p_test(dis_mean, chunk_dt1, chunk_dt2)
            else:
                pp_chunk_1 = self.p_test(dis_hh_mean, chunk_dt1, chunk_dt2)
                pp_chunk_2 = self.p_test(dis_ht_mean, chunk_dt1, chunk_dt2)
                pp_chunk_3 = self.p_test(dis_tt_mean, chunk_dt1, chunk_dt2)
                if self.ty[-1] == 'p':
                    if self.ty[:6] == 'single':
                        pp_chunk = self.r * pp_chunk_1 + (1 - self.r) * pp_chunk_2
                    elif self.ty[:6] == 'double':
                        pp_chunk = self.r * (pp_chunk_1 + pp_chunk_3) / 2 + (1 - self.r) * pp_chunk_2
                elif self.ty[-1] == 'l':
                    if self.ty[:6] == 'single':
                        pp_chunk = self.r * torch.log(self.eps+pp_chunk_1) + (1 - self.r) * torch.log(self.eps+pp_chunk_2)
                    elif self.ty[:6] == 'double':
                        pp_chunk = self.r * (torch.log(self.eps+pp_chunk_1) + torch.log(self.eps+pp_chunk_3)) / 2 + (1 - self.r) * torch.log(self.eps+pp_chunk_2)
            pp_chunks.append(pp_chunk)
        pp = torch.cat(pp_chunks, dim=0)
        return pp

    def eval(self, test_pos, test_neg, epoch):
        pos = self.get_score(test_pos)
        neg = self.get_score(test_neg)
        sorted_pos = torch.sort(pos)[0]
        sorted_neg = torch.sort(neg)[0]
        
        hit20 = eval_hits(sorted_pos, sorted_neg, 20)
        hit50 = eval_hits(sorted_pos, sorted_neg, 50)
        hit100 = eval_hits(sorted_pos, sorted_neg, 100)
        self.hits50_max = [hit50, epoch]
        self.hits100_max = [hit100, epoch]
        mrr1, mrr2 = eval_mrr(sorted_pos, sorted_neg)
        mrr = (mrr1 + mrr2) / 2
        roc_auc, pr_auc, f1 = eval_auc(sorted_pos, sorted_neg)
        
        print('hit20', hit20)
        print('hit50', hit50)
        print('hit100', hit100)
        print('roc_auc, pr_auc, f1, mrr', roc_auc, pr_auc, f1, mrr)
        

# >>> Data Loading Functions <<<
def get_train_pt(data_path, split_ratio=None, direct_load=False, dataset_name=None):
    if direct_load:
        split_edges = torch.load(os.path.join(data_path, f'{dataset_name}.pt'))
    else:
        split_edges = torch.load(os.path.join(data_path, f'split_dict_{split_ratio}.pt'))
    return split_edges['train']['edge'].to(device).to(torch.int64)
def get_test_pt(data_path, split_ratio=None, direct_load=False, dataset_name=None):
    if direct_load:
        split_edges = torch.load(os.path.join(data_path, f'{dataset_name}.pt'))
    else:
        split_edges = torch.load(os.path.join(data_path, f'split_dict_{split_ratio}.pt'))
    return split_edges['test']['edge'].to(device).to(torch.int64), split_edges['test']['edge_neg'].to(device).to(torch.int64)


# >>> Random Seed Setting <<<
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # 1. Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    
    # 2. Add arguments to the parser
    parser.add_argument("--dataset", type=str, default='pubmed')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--split_ratio", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=3)
    parser.add_argument("--h", type=int, default=12)
    parser.add_argument("--gamma", type=float, default=3)
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--c", type=int, default=1)
    parser.add_argument("--ty", type=str, default='single_d')
    parser.add_argument("--r", type=int, default=0.2)
    parser.add_argument("--neg", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--pos_ratio", type=float, default=1)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--convergence", type=float, default=0.8)
    parser.add_argument("--eval_step", type=int, default=1)
    
    # 3. Parse the command-line arguments
    args = parser.parse_args()
    
    # 4. Access the parsed arguments
    dataset_name = args.dataset
    seed = args.seed
    epoch = args.epoch
    split_ratio = args.split_ratio
    alpha = args.alpha
    h = args.h
    gamma = args.gamma
    tp = args.tp
    c = args.c
    r = args.r
    ty = args.ty
    neg = args.neg
    batch_size = args.batch_size
    pos_ratio = args.pos_ratio
    chunks = args.chunks
    convergence = args.convergence
    eval_step = args.eval_step
    direct_load = False
    
    set_random_seed(seed)
    print(f"dataset_name={dataset_name}, seed={seed}, epoch={epoch}, split_ratio={split_ratio}, alpha={alpha}, h={h}, ty={ty}, r={r}, gamma={gamma}, tp={tp}, c={c}, neg={neg}, batch_size={batch_size}, pos_ratio={pos_ratio}, chunks={chunks}, convergence={convergence}, eval_step={eval_step}")
    
    # 5. Load the dataset
    if dataset_name.startswith("ogbl_"):
        if dataset_name == "ogbl_collab":
            data_path = f'./datasets/{dataset_name}/split/time'
        elif dataset_name == "ogbl_ppa":
            data_path = f'./datasets/{dataset_name}/split/throughput'    
    elif dataset_name.startswith("ER"):
        data_path = f'./datasets/synthetic'  
        direct_load = True 
    else:
        data_path = f'./datasets/{dataset_name}/split/'
        
    print("start to load", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    train_data = get_train_pt(data_path, split_ratio, 
                            direct_load=direct_load, dataset_name=dataset_name)
    test_data = get_test_pt(data_path, split_ratio, 
                            direct_load=direct_load, dataset_name=dataset_name)
    print("end to load", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # 6. Train and Test the model
    model = CritiGraph(h=h, tp=tp, c=c, eps=1e-5, neg=neg, r=r, ty=ty, gamma=gamma, 
                        alpha=alpha, epoch=epoch, batch_size=batch_size, pos_ratio=pos_ratio,
                        chunks=chunks, convergence=convergence, eval_step=eval_step)
    model(train_data, test_data[0], test_data[1])