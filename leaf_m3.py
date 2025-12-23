from cogdl.models import BaseModel
from cogdl import experiment
import numpy as np
import pickle
import os
import torch
import warnings
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
warnings.filterwarnings('ignore', 'divide by zero encountered in log2')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        sta_ind = self.indices[idx]
        return sta_ind.clone().detach().to(dtype=torch.int64, device=device)
class CritiGraph(BaseModel):
    def __init__(self, h, tp, c, eps, neg, gamma, alpha, epoch, batch_size):
        super(CritiGraph, self).__init__()
        self.h = h
        self.tp = tp
        self.n = int(2**h)
        self.c = c
        self.k = int(c*h)
        self.eps = eps
        self.neg = neg
        self.gamma = gamma
        self.alpha = alpha
        self.epoch = epoch  
        self.batch_size = batch_size
        self.flip_masks = (1 << torch.arange(self.h, dtype=torch.int64, device=device)).unsqueeze(0).unsqueeze(2)

    def generate_distance_lookup_table(self):
        xor_results = torch.arange(self.n, dtype=torch.int64, device=device)
        return torch.where(xor_results == 0, 
                           torch.tensor(0, dtype=torch.int64, device=device), 
                           torch.floor(torch.log2(xor_results.float())) + 1).int()
    def distance(self, coord1, coord2):
        xor_result = torch.bitwise_xor(coord1, coord2)
        return torch.where(xor_result == 0, 
                           torch.tensor(0, dtype=torch.int64, device=device), 
                           torch.floor(torch.log2(xor_result.float())) + 1).int()
    def distance_test(self, coord1, coord2):
        xor_result = torch.bitwise_xor(coord1, coord2)
        return torch.where(xor_result == 0, 
                           torch.tensor(0, dtype=torch.int64), 
                           torch.floor(torch.log2(xor_result.float())) + 1).int()
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
    def p(self, dis, ig1, ig2):
        deg1, deg2 = self.degree[ig1], self.degree[ig2]        
        ap = (deg1+1)*(deg2+1)
        aa = (dis+self.eps)/torch.log(ap)[:,:,None,None]
        return 1/(1+aa**self.gamma/self.alpha)
    def p_test(self, dis, ig1, ig2):
        deg1, deg2 = self.degree_test[ig1], self.degree_test[ig2]        
        ap = (deg1+1)*(deg2+1)
        aa = (dis+self.eps)/torch.log(ap)
        return 1/(1+aa**self.gamma/self.alpha)
    def loom(self, epoch, sta_ind):
        pos_ind, lg, mask = self.neighbor_batch(sta_ind, epoch)
        sta_loc = self.locations[sta_ind]
        with torch.no_grad():
            pos_loc = torch.full((pos_ind.size(0), pos_ind.size(1), self.tp), -1, dtype=torch.int64, device=device)
            pos_loc[mask] = self.locations[pos_ind[mask]]
            cnc_loc = self.connection(self.locations[sta_ind])
            neg_ind = torch.full(pos_ind.size(), -1, dtype=torch.int64, device=device)
            neg_ind[mask] = torch.randint(0, self.num_nodes, (pos_ind.size(0), pos_ind.size(1)), dtype=torch.int64, device=device)[mask]
            neg_loc = torch.full((pos_ind.size(0), pos_ind.size(1), self.tp), -1, dtype=torch.int64, device=device)
            neg_loc[mask] = self.locations[neg_ind[mask]]
            indices = torch.randperm(cnc_loc.size(1))
            cnc_loc = cnc_loc[:, indices, :]
            mask1 = mask.unsqueeze(2).repeat(1, 1, self.tp)
            mask2 = mask1.unsqueeze(2).repeat(1, 1, cnc_loc.size(1),1)
            dis_sta_pos = torch.where(mask1, self.distance(sta_loc[:,None,:], pos_loc), -1)
            dis_sta_posum = torch.sum(dis_sta_pos, dim=-1)
            dis_sta_neg = torch.where(mask1, self.distance(sta_loc[:,None,:], neg_loc), -1)
            dis_sta_negum = torch.sum(dis_sta_neg, dim=-1)
            dis_pos_cnc = torch.where(mask2, self.distance(cnc_loc[:,None,:,:], pos_loc[:,:,None,:]), -1)
            dis_neg_cnc = torch.where(mask2, self.distance(cnc_loc[:,None,:,:], neg_loc[:,:,None,:]), -1)
            dis_new_pos = torch.where(mask2, (dis_pos_cnc-dis_sta_pos[:,:,None,:]+dis_sta_posum[:,:,None,None])/self.tp, 0)
            pos_loss = -torch.sum(torch.log(self.p(dis_new_pos, sta_ind[:,None], pos_ind)), dim=1) / lg[:,None,None]
            dis_new_neg = torch.where(mask2, (dis_neg_cnc-dis_sta_neg[:,:,None,:]+dis_sta_negum[:,:,None,None])/self.tp, 1000)
            neg_loss = -torch.sum(torch.log(1+self.eps-self.p(dis_new_neg, sta_ind[:,None], neg_ind)), dim=1) / lg[:,None,None]
            total_loss = pos_loss + neg_loss
            index = torch.argmin(total_loss, dim=1)
            i_indices, j_indices = torch.meshgrid(torch.arange(sta_ind.size(0)), torch.arange(self.tp), indexing='ij')
            self.locations[sta_ind[i_indices], j_indices] = cnc_loc[i_indices, index[i_indices, j_indices], j_indices]
            tl = torch.mean(total_loss[i_indices, index[i_indices, j_indices], j_indices])
            pl = torch.mean(pos_loss[i_indices, index[i_indices, j_indices], j_indices])
            nl = torch.mean(neg_loss[i_indices, index[i_indices, j_indices], j_indices])
        return tl, pl, nl
    def get_neighbor(self):
        neighbor = [torch.tensor(list(self.G.neighbors(ii)), dtype=torch.int64, device=device) for ii in range(self.num_nodes)]
        neighbor_dict = {ii: neighbor[ii] for ii in range(self.num_nodes)}
        neighbor_tensor = torch.full((self.num_nodes, self.max_degree), -1, dtype=torch.int64, device=device)
        for n, nbs in neighbor_dict.items():
            neighbor_tensor[n, :len(nbs)] = nbs
        
        return neighbor_dict, neighbor_tensor
    def neighbor_batch(self, sta_ind, epoch):
        bs = sta_ind.size(0)
        batch_degree = self.degree[sta_ind] # (bs)
        batch_max_degree = batch_degree.max().item()
        batch_neighbor = self.neighbor_tensor[sta_ind, :batch_max_degree] # (bs, batch_max_degree), -1 for padding
        
        if epoch > 4/5 * self.epoch:
            batch_lengths = batch_degree
            batch_mask = batch_neighbor != -1
            
        else:
            random_probs = torch.rand(bs, device=device) # (bs, )
            choosing_mask = random_probs > 0.2 # (bs, )
            batch_lengths = torch.where(choosing_mask, batch_degree, 1)
            
            one_random_neighbor = (torch.rand(bs, device=device) * batch_degree).floor().long()
            random_neighbor_mask = torch.zeros((bs, batch_max_degree), dtype=torch.bool, device=device) # (bs, max_degree)
            random_neighbor_mask[torch.arange(bs), one_random_neighbor] = True # (bs, max_degree)
            one_random_neighbor = torch.full((bs, batch_max_degree), -1, dtype=torch.int64, device=device)
            one_random_neighbor[:, 0] = batch_neighbor[random_neighbor_mask] # (bs, max_degree)            
            
            batch_neighbor = torch.where(choosing_mask.unsqueeze(1), 
                                         batch_neighbor,
                                         one_random_neighbor) # (bs, max_degree)
            batch_neighbor = batch_neighbor[:, :batch_lengths.max()]
            batch_mask = batch_neighbor != -1
        return batch_neighbor, batch_lengths, batch_mask
         
    def forward(self, graph):
        self.num_nodes = graph.num_nodes
        self.num_edges = graph.num_edges
        print('num_nodes', self.num_nodes)
        print('num_edges', self.num_edges)
        
        current_time = datetime.now()
        print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
        self.G = graph.to_networkx()
        self.degree = torch.tensor([self.G.degree(n) for n in self.G.nodes()], dtype=torch.int64, device=device)
        self.degree_test = torch.tensor([self.G.degree(n) for n in self.G.nodes()], dtype=torch.int64)
        self.max_degree = self.degree.max()
        self.neighbor, self.neighbor_tensor = self.get_neighbor()
        self.locations = torch.randint(0, self.n, (self.num_nodes, self.tp), dtype=torch.int64, device=device)
        self.li = torch.arange(self.num_nodes, dtype=torch.int64, device=device)[self.degree>0]
        dataset = MyDataset(self.li)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.get_neighbor()
        for epoch in range(self.epoch):    
            current_time = datetime.now()
            print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
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
            print(epoch, total, positive, negative)
        self.cpu()
        pid = os.getpid()
        print('pid', pid)
        torch.save(self, '/data/gc/CT/file/'+str(pid)+'.pth')
        return self.locations.cpu()
    
if __name__ == "__main__":
    current_time = datetime.now()
    print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
    model = CritiGraph(h=12, tp=16, c=1, eps=1e-5, neg=1, gamma=3, alpha=8, epoch=50, batch_size=32).to(device)
    experiment(dataset="cora", model=model, dw='embedding_link_prediction_dw', mw='embedding_link_prediction_mw')
    # experiment(dataset="citeseer", model=model, dw='embedding_link_prediction_dw', mw='embedding_link_prediction_mw', train_set_ratio=0.9)
    # experiment(dataset="pubmed", model=model, dw='embedding_link_prediction_dw', mw='embedding_link_prediction_mw', train_set_ratio=0.9)

