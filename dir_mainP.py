import numpy as np
import pickle
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
        return sta_ind.clone().detach().to(dtype=torch.int64).cuda() 
class CritiGraph(torch.nn.Module):
    def __init__(self, h, tp, c, eps, neg, v, gamma, alpha, epoch, batch_size, pos_ratio, 
                 chunks, convergence, eval_step):
        super().__init__() 
        self.h = h
        self.tp = tp
        self.n = int(2**h)
        self.c = c
        self.k = int(c*h)
        self.v = v
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
    def P(self, dis, ig1, ig2, head, tail):
        p_12 = self.p(dis, ig1, ig2)
        p_21 = self.p(dis, ig2, ig1)
        if self.v <= 0:
            P_11 = (1+self.v)*p_12*p_21+self.v*(1-p_12-p_21)*self.heaviside(p_12+p_21-1)
        else:    
            P_11 = (1-self.v)*p_12*p_21+self.v*torch.min(p_12, p_21)
        if head == 1 and tail == 1:
            # return p_12*p_21
            return P_11
        elif head == 1 and tail == 0:
            # return p_12
            return p_12 - P_11
        elif head == 0 and tail == 1:
            # return p_21
            return p_21 - P_11
        else:
            return 1-p_12-p_21+P_11
            # return (1-p_12)*(1-p_21)
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
    def loom(self, epoch, sta_ind):
        pos_ind, lg, mask = self.neighbor_batch(sta_ind, epoch)
        sta_loc = self.locations[sta_ind]
        with torch.no_grad():
            pos_loc = [torch.full((pos_ind[j].size(0), pos_ind[j].size(1), self.tp), -1, dtype=torch.int64, device=device) for j in range(4)]
            for j in range(4):
                pos_loc[j][mask[j]] = self.locations[pos_ind[j][mask[j]]]
            cnc_loc = self.connection(self.locations[sta_ind])
            neg_ind = torch.full(pos_ind[0].size(), -1, dtype=torch.int64, device=device)
            neg_ind[mask[0]] = torch.randint(0, self.num_nodes, (pos_ind[0].size(0), pos_ind[0].size(1)), dtype=torch.int64, device=device)[mask[0]]
            neg_loc = torch.full((pos_ind[0].size(0), pos_ind[0].size(1), self.tp), -1, dtype=torch.int64, device=device)
            neg_loc[mask[0]] = self.locations[neg_ind[mask[0]]]
            indices = torch.randperm(cnc_loc.size(1))
            cnc_loc = cnc_loc[:, indices, :]
            mask1 = [mask[j].unsqueeze(2).repeat(1, 1, self.tp) for j in range(4)]
            mask2 = [mask1[j].unsqueeze(2).repeat(1, 1, cnc_loc.size(1),1) for j in range(4)]
            dis_sta_pos = [torch.where(mask1[j], self.distance(sta_loc[:,None,:], pos_loc[j]), -1) for j in range(4)]
            dis_sta_posum = [torch.sum(dis_sta_pos[j], dim=-1) for j in range(4)]
            dis_sta_neg = torch.where(mask1[0], self.distance(sta_loc[:,None,:], neg_loc), -1)
            dis_sta_negum = torch.sum(dis_sta_neg, dim=-1)
            dis_pos_cnc = [torch.where(mask2[j], self.distance(cnc_loc[:,None,:,:], pos_loc[j][:,:,None,:]), -1) for j in range(4)]
            dis_neg_cnc = torch.where(mask2[0], self.distance(cnc_loc[:,None,:,:], neg_loc[:,:,None,:]), -1)
            dis_new_pos = [torch.where(mask2[j], (dis_pos_cnc[j]-dis_sta_pos[j][:,:,None,:]+dis_sta_posum[j][:,:,None,None])/self.tp, 0) for j in range(4)]
            pos_loss_01 = -torch.sum(torch.where(mask2[1], torch.log(self.eps+self.P(dis_new_pos[1], sta_ind[:,None], pos_ind[1], 0, 1)), torch.tensor(0.0, device=device)), dim=1)
            pos_loss_10 = -torch.sum(torch.where(mask2[2], torch.log(self.eps+self.P(dis_new_pos[2], sta_ind[:,None], pos_ind[2], 1, 0)), torch.tensor(0.0, device=device)), dim=1)
            pos_loss_11 = -torch.sum(torch.where(mask2[3], torch.log(self.eps+self.P(dis_new_pos[3], sta_ind[:,None], pos_ind[3], 1, 1)), torch.tensor(0.0, device=device)), dim=1)
            # pos_loss = (pos_loss_01 + pos_loss_10 + 2*pos_loss_11) / lg[0][:,None,None]
            lg1 = torch.where(lg[1]==0, torch.tensor(1.0, device=device), lg[1])[:,None,None]
            lg2 = torch.where(lg[2]==0, torch.tensor(1.0, device=device), lg[2])[:,None,None]
            lg3 = torch.where(lg[3]==0, torch.tensor(1.0, device=device), lg[3])[:,None,None]
            pos_loss = pos_loss_01 / lg1 + pos_loss_10 / lg2 + 2*pos_loss_11 / lg3
            dis_new_neg = torch.where(mask2[0], (dis_neg_cnc-dis_sta_neg[:,:,None,:]+dis_sta_negum[:,:,None,None])/self.tp, 1000)
            neg_loss = -torch.sum(torch.where(mask2[0], torch.log(self.eps+self.P(dis_new_neg, sta_ind[:,None], neg_ind, 0, 0)), torch.tensor(0.0, device=device)), dim=1) / lg[0][:,None,None]
            total_loss = self.pos_ratio * pos_loss + neg_loss
            index = torch.argmin(total_loss, dim=1)
            i_indices, j_indices = torch.meshgrid(torch.arange(sta_ind.size(0)), torch.arange(self.tp), indexing='ij')
            self.locations[sta_ind[i_indices], j_indices] = cnc_loc[i_indices, index[i_indices, j_indices], j_indices]
            tl = torch.mean(total_loss[i_indices, index[i_indices, j_indices], j_indices])
            pl = torch.mean(pos_loss[i_indices, index[i_indices, j_indices], j_indices])
            nl = torch.mean(neg_loss[i_indices, index[i_indices, j_indices], j_indices])
        return tl, pl, nl

    def get_neighbor(self):
        neighbor_22 = [torch.tensor(list(set(self.G.predecessors(ii)) | set(self.G.successors(ii))), dtype=torch.int64, device=device) 
            for ii in range(self.num_nodes)]
        neighbor_01 = [torch.tensor(list(set(self.G.predecessors(ii)) - set(self.G.successors(ii))), dtype=torch.int64, device=device)
            for ii in range(self.num_nodes)]
        neighbor_10 = [torch.tensor(list(set(self.G.successors(ii)) - set(self.G.predecessors(ii))), dtype=torch.int64, device=device)
            for ii in range(self.num_nodes)]
        neighbor_11 = [torch.tensor(list(set(self.G.predecessors(ii)) & set(self.G.successors(ii))), dtype=torch.int64, device=device)
            for ii in range(self.num_nodes)]
        neighbor = [neighbor_22, neighbor_01, neighbor_10, neighbor_11]
        neighbor_dict = [{ii: neighbor[j][ii] for ii in range(self.num_nodes)} for j in range(4)]
        neighbor_tensor = [torch.full((self.num_nodes, self.max_degree), -1, dtype=torch.int64, device=device) for _ in range(4)]
        for j in range(4):
            for n, nbs in neighbor_dict[j].items():
                neighbor_tensor[j][n, :len(nbs)] = nbs
        return neighbor_tensor
    def neighbor_batch(self, sta_ind, epoch):
        bs = sta_ind.size(0)
        batch_max_degree = self.degree[sta_ind].max().item()
        batch_neighbor = [self.neighbor_tensor[j][sta_ind, :batch_max_degree] for j in range(4)]
        batch_lengths = [self.degree[sta_ind], 
                        self.in_degree[sta_ind] - self.degree_11[sta_ind],
                        self.out_degree[sta_ind] - self.degree_11[sta_ind],
                        self.degree_11[sta_ind]]
        if epoch <= self.convergence * self.epoch:
            random_probs = torch.rand(bs, device=device) # (bs, )
            choosing_mask = random_probs > 0.2 # (bs, )
            batch_lengths = [torch.where(choosing_mask, batch_lengths[j], 1) for j in range(4)]
            
            one_random_neighbor = [(torch.rand(bs, device=device) * batch_lengths[j]).floor().long() for j in range(4)] # (bs, )
            random_neighbor_mask = [torch.zeros((bs, batch_max_degree), dtype=torch.bool, device=device) for _ in range(4)] # (bs, max_degree)
            for j in range(4):
                random_neighbor_mask[j][torch.arange(bs), one_random_neighbor[j]] = True # (bs, max_degree)
                one_random_neighbor[j] = torch.full((bs, batch_max_degree), -1, dtype=torch.int64, device=device)
                one_random_neighbor[j][:, 0] = batch_neighbor[j][random_neighbor_mask[j]] # (bs, max_degree)            
                batch_neighbor[j] = torch.where(choosing_mask.unsqueeze(1), 
                                            batch_neighbor[j],
                                            one_random_neighbor[j]) # (bs, max_degree)
                batch_neighbor[j] = batch_neighbor[j][:, :batch_lengths[j].max()]
        batch_mask = [batch_neighbor[j] != -1 for j in range(4)]
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
        self.out_degree = torch.IntTensor([self.G.out_degree(n) for n in self.G.nodes()]).cuda().to(torch.int64)
        self.in_degree = torch.IntTensor([self.G.in_degree(n) for n in self.G.nodes()]).cuda().to(torch.int64)
        self.degree_11 = torch.IntTensor([(len(list(set(self.G.predecessors(n)) & set(self.G.successors(n))))) for n in self.G.nodes()]).cuda().to(torch.int64)
        self.degree = torch.IntTensor([self.G.degree(n) for n in self.G.nodes()]).cuda().to(torch.int64) - self.degree_11
        self.max_degree = self.degree.max()
        self.neighbor_tensor = self.get_neighbor()
        self.locations = torch.randint(0, self.n, (self.num_nodes, self.tp), dtype=torch.int64, device=device)
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
            chunk_vector1 = self.locations[chunk_dt1]
            chunk_vector2 = self.locations[chunk_dt2]
            dis = self.distance(chunk_vector1, chunk_vector2).float()
            dis_mean = torch.mean(dis, dim=-1)
            pp_chunk = self.p_test(dis_mean, chunk_dt1, chunk_dt2)
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
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument("--h", type=int, default=12)
    parser.add_argument("--gamma", type=float, default=3)
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--c", type=int, default=1)
    parser.add_argument("--v", type=float, default=0.8)
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
    neg = args.neg
    v = args.v
    batch_size = args.batch_size
    pos_ratio = args.pos_ratio
    chunks = args.chunks
    convergence = args.convergence
    eval_step = args.eval_step
    direct_load = False
    
    set_random_seed(seed)
    print(f"dataset_name={dataset_name}, seed={seed}, epoch={epoch}, split_ratio={split_ratio}, alpha={alpha}, h={h}, gamma={gamma}, tp={tp}, c={c}, neg={neg}, v={v}, batch_size={batch_size}, pos_ratio={pos_ratio}, chunks={chunks}, convergence={convergence}, eval_step={eval_step}")
    
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
    model = CritiGraph(h=h, tp=tp, c=c, eps=1e-5, neg=neg, v=v, gamma=gamma, 
                        alpha=alpha, epoch=epoch, batch_size=batch_size, pos_ratio=pos_ratio,
                        chunks=chunks, convergence=convergence, eval_step=eval_step)
    model(train_data, test_data[0], test_data[1])
    

    