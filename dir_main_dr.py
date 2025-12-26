import numpy as np
import os
import torch
import warnings
from datetime import datetime
from torch.utils.data import Dataset
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



class MyDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        sta_ind = self.indices[idx]
        return sta_ind.clone().detach().to(dtype=torch.int64).cuda()


# >>> Model <<< #
class CritiGraph(torch.nn.Module):
    def __init__(self, h, tp, c, eps, neg, gamma, alpha, epoch, batch_size, pos_ratio,
                 chunks, convergence, eval_step, oi):
        super().__init__()
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
        self.batch_edge = batch_size

        self.pos_ratio = pos_ratio
        self.flip_masks = (1 << torch.arange(self.h, dtype=torch.int64, device=device)).unsqueeze(0).unsqueeze(2)
        self.distance_lookup_table = self.generate_distance_lookup_table()
        self.chunks = chunks
        self.convergence = convergence
        self.eval_step = eval_step
        self.oi = oi  

    
    def generate_distance_lookup_table(self):
        xor_results = torch.arange(self.n, dtype=torch.int64, device=device)
        return torch.where(
            xor_results == 0,
            torch.tensor(0, dtype=torch.int64, device=device),
            torch.floor(torch.log2(xor_results.float())) + 1
        ).int()

    def distance(self, coord1, coord2):
        xor_result = torch.bitwise_xor(coord1, coord2)
        return self.distance_lookup_table[xor_result]

    def generate_random_masks(self, sz):
        upper_bounds = 2**torch.arange(self.h, dtype=torch.int64, device=device)
        random_numbers = torch.randint(
            0, self.n, (self.h, sz, self.k, self.tp),
            dtype=torch.int64, device=device
        )
        masks = random_numbers % upper_bounds.view(-1, 1, 1, 1)
        return masks.permute(1, 0, 2, 3)

    def connection(self, ori_int):
        # ori_int: (B,tp)
        flipped_ints = ori_int.unsqueeze(1) ^ self.flip_masks          # (B,h,tp)
        random_masks = self.generate_random_masks(flipped_ints.size(0)) # (B,h,k,tp)
        result = (flipped_ints.unsqueeze(2) ^ random_masks).view(
            flipped_ints.size(0), self.h * self.k, self.tp
        )                                                               # (B,h*k,tp)
        return torch.cat((result, ori_int.unsqueeze(1)), dim=1)         # (B,h*k+1,tp)

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
            return 1 - p_12
        elif key == 3:
            return 1 - p_21

    def p(self, dis, ig1, ig2):
        if self.oi:
            deg1, deg2 = self.out_degree[ig1], self.in_degree[ig2]
        else:
            deg1, deg2 = self.degree[ig1], self.degree[ig2]
        ap = (deg1+1)*(deg2+1)
        aa = (dis+self.eps)/torch.log(ap)[:,:,None,None]
        return 1/(1+aa**self.gamma/self.alpha)

    def p_test(self, dis, ig1, ig2):
        if self.oi:
            deg1, deg2 = self.out_degree[ig1], self.in_degree[ig2]
        else:
            deg1, deg2 = self.degree[ig1], self.degree[ig2]
        ap = (deg1+1)*(deg2+1)
        aa = (dis+self.eps)/torch.log(ap)
        return 1/(1+aa**self.gamma/self.alpha)

    
    def _p_flat(self, dis: torch.Tensor, ig1: torch.Tensor, ig2: torch.Tensor) -> torch.Tensor:
        """
         p， dis  (E,L,tp)  (E,tp)。
        ig1, ig2:  (E,) （）。
        """
        if self.oi:
            deg1 = self.out_degree[ig1]
            deg2 = self.in_degree[ig2]
        else:
            deg1 = self.degree[ig1]
            deg2 = self.degree[ig2]
        ap = (deg1 + 1) * (deg2 + 1)  # (E,)
        denom = torch.log(ap.float()).view(-1, *([1] * (dis.dim() - 1)))
        aa = (dis + self.eps) / denom
        return 1.0 / (1.0 + (aa ** self.gamma) / self.alpha)

    def _P_flat(self, dis: torch.Tensor, ig1: torch.Tensor, ig2: torch.Tensor, key: int) -> torch.Tensor:
        """
         P ：
        key = 0 -> p_12
        key = 1 -> p_21
        key = 2 -> 1 - p_12
        key = 3 -> 1 - p_21
        """
        p_12 = self._p_flat(dis, ig1, ig2)
        p_21 = self._p_flat(dis, ig2, ig1)
        if key == 0:
            return p_12
        elif key == 1:
            return p_21
        elif key == 2:
            return 1.0 - p_12
        elif key == 3:
            return 1.0 - p_21
        else:
            raise ValueError(f"Unknown key={key}")

    
    def get_neighbor(self):
        """
         self.G  CSR :
            out:  =  u,  =  v,  u->v
            in :  =  u,  =  v,  v->u
        """
        edges = torch.tensor(list(self.G.edges()), dtype=torch.long, device=device)
        if edges.numel() == 0:
            self.rowptr_out = torch.zeros(self.num_nodes + 1, dtype=torch.long, device=device)
            self.col_out = torch.tensor([-1], dtype=torch.long, device=device)
            self.rowptr_in = torch.zeros(self.num_nodes + 1, dtype=torch.long, device=device)
            self.col_in = torch.tensor([-1], dtype=torch.long, device=device)
            return

        src = edges[:, 0]
        dst = edges[:, 1]

        idx_out = torch.stack((src, dst), dim=0)  # (2,E)
        idx_in  = torch.stack((dst, src), dim=0)  # (2,E)

        def get_sparse_adj(idx: torch.Tensor):
            val = torch.ones(idx.size(1), dtype=torch.bool, device=device)
            adj = torch.sparse_coo_tensor(
                indices=idx,
                values=val,
                size=(self.num_nodes, self.num_nodes)
            ).coalesce()
            return adj

        adj_out = get_sparse_adj(idx_out)
        adj_in  = get_sparse_adj(idx_in)

        def coo_to_csr(adj: torch.Tensor):
            row, col = adj.indices()  
            col = torch.cat((col, torch.tensor([-1], dtype=torch.long, device=device)))
            counts = torch.bincount(row, minlength=self.num_nodes)
            rowptr = torch.empty(self.num_nodes + 1, dtype=torch.long, device=device)
            rowptr[0] = 0
            rowptr[1:] = torch.cumsum(counts, dim=0)
            return rowptr, col

        self.rowptr_out, self.col_out = coo_to_csr(adj_out)
        self.rowptr_in,  self.col_in  = coo_to_csr(adj_in)

    
    @torch.no_grad()
    def neighbor_batch_csr(self, sta_ind: torch.Tensor, choosing_mask_b: torch.Tensor):
        """
         CSR  batch  out / in ， choosing_mask_b=False “”。

        :
            sta_ind        : (B,) int64, batch 
            choosing_mask_b: (B,) bool
        :
            src_out, dst_out, cnt_out,
            src_in,  dst_in,  cnt_in
        :
            src_* : (E_*,)  batch  0..B-1
            dst_* : (E_*,) 
            cnt_* : (B,)  batch “”（， 1）
        """
        assert sta_ind.dim() == 1 and sta_ind.dtype == torch.int64
        assert choosing_mask_b.dim() == 1 and choosing_mask_b.dtype == torch.bool
        B = sta_ind.numel()
        device_local = sta_ind.device

        def expand_and_sample(ro: torch.Tensor, co: torch.Tensor):
            ro = ro
            co = co[:-1]  
            off = ro[sta_ind]                     # (B,)
            deg = ro[sta_ind + 1] - off           # (B,)
            valid = deg > 0                       # (B,)

            if valid.any():
                sta_valid = torch.arange(B, device=device_local, dtype=torch.long)[valid]
                lenv = deg[valid]                                      # (B_valid,)
                src_flat = torch.repeat_interleave(sta_valid, lenv)    # (E,)
                base = off[valid].repeat_interleave(lenv)              # (E,)

                head = torch.cumsum(lenv, dim=0) - lenv                # (B_valid,)
                ofs = torch.arange(lenv.sum().item(), device=device_local, dtype=torch.long) - \
                      torch.repeat_interleave(head, lenv)              # (E,)

                dst_flat = co[base + ofs]                              # (E,)
                bc = torch.bincount(src_flat, minlength=B)             # (B,)

                
                if (~choosing_mask_b[src_flat]).any():
                    keep_all = choosing_mask_b[src_flat]               # (E,)
                    head_e = torch.cumsum(bc, 0) - bc                  # (B,)
                    idx_in_seg = torch.arange(src_flat.numel(), device=device_local, dtype=torch.long) - head_e[src_flat]
                    seg_len = bc.clamp_min(1)                          # (B,)
                    pick = (torch.rand(B, device=device_local) * seg_len.float()).floor().to(torch.long)
                    keep = keep_all | (idx_in_seg == pick[src_flat])
                    src_flat = src_flat[keep]
                    dst_flat = dst_flat[keep]
                    bc = torch.bincount(src_flat, minlength=B)
            else:
                src_flat = torch.empty(0, dtype=torch.long, device=device_local)
                dst_flat = torch.empty(0, dtype=torch.long, device=device_local)
                bc = torch.zeros(B, dtype=torch.long, device=device_local)

            return src_flat, dst_flat, bc

        src_out, dst_out, bc_out = expand_and_sample(self.rowptr_out, self.col_out)
        src_in,  dst_in,  bc_in  = expand_and_sample(self.rowptr_in,  self.col_in)

        cnt_out = bc_out.clamp_min(1)
        cnt_in  = bc_in.clamp_min(1)

        return src_out, dst_out, cnt_out, src_in, dst_in, cnt_in

    
    def edge_capped_node_batches(self, epoch: int, num_edges_cap: int, shuffle: bool = True):
        """
         num_edges_cap ， batch “”
            E_batch := E_out + E_in <= num_edges_cap。

        epoch <= convergence * epoch ：
            ~20%  out / in  1 ；
        ：
             out / in 。

        :
            (sta_ind_b, choosing_mask_b)  GPU。
        """
        device_local = self.out_degree.device

        if epoch <= self.convergence * self.epoch:
            choosing_mask_global = (torch.rand(self.num_nodes, device=device_local) > 0.2)
        else:
            choosing_mask_global = torch.ones(self.num_nodes, dtype=torch.bool, device=device_local)

        
        nodes = self.li.clone()
        if shuffle:
            perm = torch.randperm(nodes.numel(), device=device_local)
            nodes = nodes[perm]

        deg_out = self.out_degree[nodes]
        deg_in  = self.in_degree[nodes]
        mask_b  = choosing_mask_global[nodes]

        eff_out = torch.where(mask_b, deg_out, deg_out.clamp_max(1))
        eff_in  = torch.where(mask_b, deg_in,  deg_in.clamp_max(1))
        cost    = eff_out + eff_in   

        cur_nodes = []
        cur_mask  = []
        cur_edges = 0

        cost_cpu  = cost.detach().cpu().tolist()
        nodes_cpu = nodes.detach().cpu().tolist()
        mask_cpu  = mask_b.detach().cpu().tolist()

        for n_id, c, m in zip(nodes_cpu, cost_cpu, mask_cpu):
            c = int(c)
            if c == 0:
                continue

            if len(cur_nodes) > 0 and (cur_edges + c) > num_edges_cap:
                sta_ind_b = torch.tensor(cur_nodes, device=device_local, dtype=torch.long)
                choosing_mask_b = torch.tensor(cur_mask, device=device_local, dtype=torch.bool)
                yield sta_ind_b, choosing_mask_b
                cur_nodes, cur_mask, cur_edges = [], [], 0

            cur_nodes.append(n_id)
            cur_mask.append(bool(m))
            cur_edges += c

            if cur_edges > num_edges_cap:
                sta_ind_b = torch.tensor(cur_nodes, device=device_local, dtype=torch.long)
                choosing_mask_b = torch.tensor(cur_mask, device=device_local, dtype=torch.bool)
                yield sta_ind_b, choosing_mask_b
                cur_nodes, cur_mask, cur_edges = [], [], 0

        if len(cur_nodes) > 0:
            sta_ind_b = torch.tensor(cur_nodes, device=device_local, dtype=torch.long)
            choosing_mask_b = torch.tensor(cur_mask, device=device_local, dtype=torch.bool)
            yield sta_ind_b, choosing_mask_b

    
    @torch.no_grad()
    def loom_v2(self, epoch: int, sta_ind: torch.Tensor, choosing_mask_b: torch.Tensor):
        """
         CSR  + ， head / tail  embedding。
        """
        device_local = sta_ind.device
        B = sta_ind.size(0)
        tp = self.tp

        
        sta_head = self.locations[0][sta_ind]   # (B,tp)
        sta_tail = self.locations[1][sta_ind]   # (B,tp)

        
        cnc_head = self.connection(sta_head)    # (B,L,tp)
        cnc_tail = self.connection(sta_tail)    # (B,L,tp)
        L = cnc_head.size(1)

        perm_head = torch.randperm(L, device=device_local)
        perm_tail = torch.randperm(L, device=device_local)
        cnc_head = cnc_head[:, perm_head, :]
        cnc_tail = cnc_tail[:, perm_tail, :]

        
        src_out, dst_out, cnt_out, src_in, dst_in, cnt_in = self.neighbor_batch_csr(sta_ind, choosing_mask_b)

        
        def branch(sta_loc, cnc_loc, loc_other, src_flat, dst_flat, cnt, key_pos, key_neg):
            """
            sta_loc : (B,tp)， embedding（head  tail）
            cnc_loc : (B,L,tp)，
            loc_other: (num_nodes,tp)， embedding（head<->tail）
            src_flat: (E,) batch  [0..B-1]
            dst_flat: (E,) 
            cnt     : (B,) （，>=1）
            key_pos :  P  key (0 or 1)
            key_neg :  P  key (2 or 3)
            :
                pos_loss, neg_loss: (B,L,tp)
            """
            pos_loss = torch.zeros((B, L, tp), dtype=torch.float32, device=device_local)
            neg_loss = torch.zeros((B, L, tp), dtype=torch.float32, device=device_local)

            E = src_flat.numel()
            if E == 0:
                return pos_loss, neg_loss

            sta_src_vec = sta_loc[src_flat]          # (E,tp)
            cnc_src_vec = cnc_loc[src_flat]          # (E,L,tp)
            pos_dst_vec = loc_other[dst_flat]        # (E,tp)

            
            dis_sta_pos = self.distance(sta_src_vec, pos_dst_vec).float()      # (E,tp)
            dis_sta_posum = dis_sta_pos.sum(dim=-1)                             # (E,)
            dis_pos_cnc = self.distance(cnc_src_vec, pos_dst_vec.unsqueeze(1)).float()  # (E,L,tp)
            dis_new_pos = (dis_pos_cnc
                           - dis_sta_pos.unsqueeze(1)
                           + dis_sta_posum.view(-1, 1, 1)) / tp                # (E,L,tp)

            ig1_pos = sta_ind[src_flat]     # (E,)
            ig2_pos = dst_flat              # (E,)
            P_pos = self._P_flat(dis_new_pos, ig1_pos, ig2_pos, key_pos)       # (E,L,tp)
            ll_pos = -torch.log(self.eps + P_pos)                              # (E,L,tp)

            pos_acc = torch.zeros((B, L, tp), dtype=torch.float32, device=device_local)
            pos_acc.index_add_(0, src_flat, ll_pos)
            pos_loss = pos_acc / cnt.view(-1, 1, 1)                            # (B,L,tp)

            
            neg_dst_flat = torch.randint(
                0, self.num_nodes, (E,),
                dtype=torch.long, device=device_local
            )
            neg_dst_vec = loc_other[neg_dst_flat]                              # (E,tp)

            dis_sta_neg = self.distance(sta_src_vec, neg_dst_vec).float()      # (E,tp)
            dis_sta_negum = dis_sta_neg.sum(dim=-1)                            # (E,)
            dis_neg_cnc = self.distance(cnc_src_vec, neg_dst_vec.unsqueeze(1)).float()  # (E,L,tp)
            dis_new_neg = (dis_neg_cnc
                           - dis_sta_neg.unsqueeze(1)
                           + dis_sta_negum.view(-1, 1, 1)) / tp                # (E,L,tp)

            ig1_neg = ig1_pos
            ig2_neg = neg_dst_flat
            P_neg = self._P_flat(dis_new_neg, ig1_neg, ig2_neg, key_neg)       # (E,L,tp)
            ll_neg = -torch.log(self.eps + P_neg)                              # (E,L,tp)

            neg_acc = torch.zeros((B, L, tp), dtype=torch.float32, device=device_local)
            neg_acc.index_add_(0, src_flat, ll_neg)
            neg_loss = neg_acc / cnt.view(-1, 1, 1)                            # (B,L,tp)

            return pos_loss, neg_loss

        
        pos_out, neg_out = branch(
            sta_head, cnc_head, self.locations[1],
            src_out, dst_out, cnt_out,
            key_pos=0, key_neg=2
        )
        
        pos_in, neg_in = branch(
            sta_tail, cnc_tail, self.locations[0],
            src_in, dst_in, cnt_in,
            key_pos=1, key_neg=3
        )

        total_out = self.pos_ratio * pos_out + neg_out   # (B,L,tp)
        total_in  = self.pos_ratio * pos_in  + neg_in    # (B,L,tp)

        index_out = torch.argmin(total_out, dim=1)       # (B,tp)
        index_in  = torch.argmin(total_in,  dim=1)       # (B,tp)

        b_idx = torch.arange(B, device=device_local)[:, None]       # (B,1)
        t_idx = torch.arange(tp, device=device_local)[None, :]      # (1,tp)

        
        self.locations[0][sta_ind[b_idx], t_idx] = cnc_head[b_idx, index_out, t_idx]
        self.locations[1][sta_ind[b_idx], t_idx] = cnc_tail[b_idx, index_in,  t_idx]

        def gather_bt(M, idx):
            return M[b_idx, idx, t_idx].mean()

        tl = gather_bt(total_out, index_out) + gather_bt(total_in, index_in)
        pl = gather_bt(pos_out,   index_out) + gather_bt(pos_in,   index_in)
        nl = gather_bt(neg_out,   index_out) + gather_bt(neg_in,   index_in)
        return tl, pl, nl

    
    def forward(self, graph, test_pos, test_neg):
        current_time = datetime.now()
        print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
        print("start to load data")

        self.Go = nx.DiGraph()
        for edge in graph:
            self.Go.add_edge(edge[0].item(), edge[1].item())

        old_nodes = list(self.Go.nodes())
        self.mapping = {old_node: new_node for new_node, old_node in enumerate(old_nodes)}

        num_old_nodes = max(old_nodes) + 10  
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

        if MRR_NODE.run:
            MRR_NODE.init(self.num_nodes, block_size=16384, neg_ratio=1000)

        self.out_degree = torch.IntTensor([self.G.out_degree(n) for n in self.G.nodes()]).cuda().to(torch.int64)
        self.in_degree  = torch.IntTensor([self.G.in_degree(n)  for n in self.G.nodes()]).cuda().to(torch.int64)
        self.degree     = torch.IntTensor([self.G.degree(n)     for n in self.G.nodes()]).cuda().to(torch.int64)
        self.max_degree = self.degree.max()

        
        self.get_neighbor()

        
        self.locations_head = torch.randint(0, self.n, (self.num_nodes, self.tp), dtype=torch.int64, device=device)
        self.locations_tail = torch.randint(0, self.n, (self.num_nodes, self.tp), dtype=torch.int64, device=device)
        self.locations = [self.locations_head, self.locations_tail]

        
        self.li = torch.arange(self.num_nodes, dtype=torch.int64, device=device)[
            (self.out_degree + self.in_degree) > 0
        ]

        for epoch in range(self.epoch):
            print("current_time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "pos+neg, pos, neg")
            sys.stdout.flush()

            total_wsum = 0.0
            pos_wsum   = 0.0
            neg_wsum   = 0.0
            node_count = 0
            total_time = 0.0

            
            for sta_ind_b, choosing_mask_b in self.edge_capped_node_batches(epoch, self.batch_edge, shuffle=True):
                st_time = datetime.now()
                tl, pl, nl = self.loom_v2(epoch, sta_ind_b, choosing_mask_b)
                B = sta_ind_b.numel()
                total_wsum += tl.item() * B
                pos_wsum   += pl.item() * B
                neg_wsum   += nl.item() * B
                node_count += B
                ed_time = datetime.now()
                total_time += (ed_time - st_time).total_seconds()

            print("Finished training epoch:", epoch, "total time:", total_time)
            
            if node_count == 0:
                total = pos = neg = 0.0
            else:
                total = total_wsum / node_count
                pos   = pos_wsum   / node_count
                neg   = neg_wsum   / node_count

            print(epoch, float(total), float(pos), float(neg))

            if (epoch + 1) % self.eval_step == 0 or epoch == self.epoch - 1:
                print("current_time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print("start to eval")
                sys.stdout.flush()
                self.eval(test_pos, test_neg, epoch)

    
    def get_score(self, data):
        dt = data.t()
        dt1 = self.bucket_tensor[dt[0]]
        dt2 = self.bucket_tensor[dt[1]]
        N = dt1.shape[0]
        chunk_size = (N + self.chunks - 1) // self.chunks
        pp_chunks = []
        for i in range(self.chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, N)
            if start >= end:
                break
            chunk_dt1 = dt1[start:end]
            chunk_dt2 = dt2[start:end]
            chunk_vector1 = self.locations[0][chunk_dt1]
            chunk_vector2 = self.locations[1][chunk_dt2]
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
        if MRR_NODE.run:
            mrr_opt, mrr_pess = MRR_NODE.evaluate(test_pos, pos, self.get_score)
        else:
            mrr_opt, mrr_pess = eval_mrr(sorted_pos, sorted_neg)
        roc_auc, pr_auc, f1 = eval_auc(sorted_pos, sorted_neg)

        print('hit20', hit20)
        print('hit50', hit50)
        print('hit100', hit100)
        print('roc_auc, pr_auc, f1, mrr_opt, mrr_pess', roc_auc, pr_auc, f1, mrr_opt, mrr_pess)



# >>> Data Loading Functions <<<
def get_train_pt(data_path, split_ratio=None, direct_load=False, dataset_name=None):
    if direct_load:
        split_edges = torch.load(data_path)
    else:
        split_edges = torch.load(os.path.join(data_path, f'split_dict_{split_ratio}.pt'))
    return split_edges['train']['edge'].to(device).to(torch.int64)
def get_test_pt(data_path, split_ratio=None, direct_load=False, dataset_name=None):
    if direct_load:
        split_edges = torch.load(data_path)
    else:
        split_edges = torch.load(os.path.join(data_path, f'split_dict_{split_ratio}.pt'))
    return split_edges['test']['edge'].to(device).to(torch.int64), split_edges['test']['edge_neg'].to(device).to(torch.int64)


# >>> Random Seed Setting <<< #
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
    parser.add_argument("--dataset", type=str, default='cora_ml')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=3)
    parser.add_argument("--h", type=int, default=12)
    parser.add_argument("--gamma", type=float, default=3)
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--c", type=int, default=1)
    parser.add_argument("--neg", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)  
    parser.add_argument("--pos_ratio", type=float, default=1)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--convergence", type=float, default=0.8)
    parser.add_argument("--eval_step", type=int, default=1)
    parser.add_argument("--oi", type=int, default=1)
    parser.add_argument("--save_score", type=str, default=None)
    parser.add_argument("--save_emb", type=str, default=None)

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
    batch_size = args.batch_size
    pos_ratio = args.pos_ratio
    chunks = args.chunks
    convergence = args.convergence
    eval_step = args.eval_step
    direct_load = False
    oi = args.oi
    save_score = args.save_score
    save_emb = args.save_emb
    assert oi == 0 or oi == 1, "oi should be 0 or 1"

    set_random_seed(seed)
    print(f"dataset_name={dataset_name}, seed={seed}, epoch={epoch}, split_ratio={split_ratio}, "
          f"alpha={alpha}, h={h}, gamma={gamma}, tp={tp}, c={c}, neg={neg}, batch_size={batch_size}, "
          f"pos_ratio={pos_ratio}, chunks={chunks}, convergence={convergence}, eval_step={eval_step}, oi={oi}")

    # 5. Load the dataset
    if dataset_name == 'ogbl_citation2':
        MRR_NODE.activate()
        data_path = f'./dir_datasets/{dataset_name}/split/'
    elif dataset_name.startswith("ER"):
        data_path = f'./dir_datasets/{dataset_name}/data.pt'  
        direct_load = True 
    else:
        data_path = f'./dir_datasets/{dataset_name}/split/'


    print("start to load", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    train_data = get_train_pt(
        data_path, split_ratio,
        direct_load=direct_load, dataset_name=dataset_name
    )
    test_data = get_test_pt(
        data_path, split_ratio,
        direct_load=direct_load, dataset_name=dataset_name
    )
    print("end to load", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # 6. Train and Test the model
    model = CritiGraph(
        h=h, tp=tp, c=c, eps=1e-5, neg=neg, gamma=gamma,
        alpha=alpha, epoch=epoch, batch_size=batch_size, pos_ratio=pos_ratio,
        chunks=chunks, convergence=convergence, eval_step=eval_step,
        oi=oi
    )
    model(train_data, test_data[0], test_data[1])

    if save_emb:
        torch.save(
            {
                'locations': [model.locations[0].cpu(), model.locations[1].cpu()],
                'table': model.distance_lookup_table.cpu(),
                'bucket': model.bucket_tensor.cpu(),
                'in_degree': model.in_degree.cpu(),
                'out_degree': model.out_degree.cpu(),
                'gamma': model.gamma,
                'alpha': model.alpha,
                'eps': model.eps,
            },
            save_emb
        )
