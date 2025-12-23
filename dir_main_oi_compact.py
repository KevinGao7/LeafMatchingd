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
from typing import Optional, Tuple, List

# >>> Dataset and Model <<< #
class MyDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices.cuda().to(dtype=torch.int64)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        sta_ind = self.indices[idx]
        return sta_ind 
    
class CritiGraph(torch.nn.Module):
    def __init__(self, h, tp, c, eps, neg, gamma, alpha, epoch, batch_size, pos_ratio, 
                 chunks, convergence, eval_step, fixed_L):
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
        self.pos_ratio = pos_ratio
        self.flip_masks = (1 << torch.arange(self.h, dtype=torch.int64, device=device)).unsqueeze(0).unsqueeze(2)
        self.distance_lookup_table = self.generate_distance_lookup_table()
        self.chunks = chunks
        self.convergence = convergence
        self.eval_step = eval_step
        self.fixed_L = fixed_L
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
        if key == 'in':
            return p_21
        elif key == 'out':
            return p_12
        elif key == 'neg':
            return (1-p_12)*(1-p_21)
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
        r1 = torch.rand(sta_ind.shape[0], device=device)
        r2 = [torch.rand(sta_ind.shape[0], device=device) for _ in range(3)]
        # pos_ind, lg, mask, max_degree = self.neighbor_batch(sta_ind, epoch, r1=r1, r2=r2)
        pos_ind, lg, mask = self.neighbor_batch_fixed(sta_ind, epoch, r1=r1, r2=r2) # type: ignore
        # self.debug_check_neighbors("all", 0, sta_ind, pos_ind[0], mask[0], lg[0])
        # self.debug_check_neighbors("in" , 1, sta_ind, pos_ind[1], mask[1], lg[1])
        # self.debug_check_neighbors("out", 2, sta_ind, pos_ind[2], mask[2], lg[2])

        # ok, _ = self.check_neighbor_batch_outputs_equal_setwise(
        #     pos_ind, lg, mask,
        #     pos_ind_new, lg_new, mask_new
        # )
        
        # if not ok:
        #     self.debug_one_row(sta_ind, 53, pos_ind, mask, pos_ind_new, mask_new, j=0)
        #     raise ValueError("neighbor_batch outputs are not consistent with neighbor_batch_fixed outputs.")
        
        sta_loc = self.locations[sta_ind]
        with torch.no_grad():
            pos_loc = [torch.full((pos_ind[j].size(0), pos_ind[j].size(1), self.tp), -1, dtype=torch.int64, device=device) for j in range(3)]
            for j in range(3):
                pos_loc[j][mask[j]] = self.locations[pos_ind[j][mask[j]]]
            cnc_loc = self.connection(self.locations[sta_ind])
            neg_ind = torch.full(pos_ind[0].size(), -1, dtype=torch.int64, device=device)
            neg_ind[mask[0]] = torch.randint(0, self.num_nodes, (pos_ind[0].size(0), pos_ind[0].size(1)), dtype=torch.int64, device=device)[mask[0]]
            neg_loc = torch.full((pos_ind[0].size(0), pos_ind[0].size(1), self.tp), -1, dtype=torch.int64, device=device)
            neg_loc[mask[0]] = self.locations[neg_ind[mask[0]]]
            indices = torch.randperm(cnc_loc.size(1))
            cnc_loc = cnc_loc[:, indices, :]
            mask1 = [mask[j].unsqueeze(2).repeat(1, 1, self.tp) for j in range(3)]
            mask2 = [mask1[j].unsqueeze(2).repeat(1, 1, cnc_loc.size(1),1) for j in range(3)]
            dis_sta_pos = [torch.where(mask1[j], self.distance(sta_loc[:,None,:], pos_loc[j]), -1) for j in range(3)]
            dis_sta_posum = [torch.sum(dis_sta_pos[j], dim=-1) for j in range(3)]
            dis_sta_neg = torch.where(mask1[0], self.distance(sta_loc[:,None,:], neg_loc), -1)
            dis_sta_negum = torch.sum(dis_sta_neg, dim=-1)
            dis_pos_cnc = [torch.where(mask2[j], self.distance(cnc_loc[:,None,:,:], pos_loc[j][:,:,None,:]), -1) for j in range(3)]
            dis_neg_cnc = torch.where(mask2[0], self.distance(cnc_loc[:,None,:,:], neg_loc[:,:,None,:]), -1)
            dis_new_pos = [torch.where(mask2[j], (dis_pos_cnc[j]-dis_sta_pos[j][:,:,None,:]+dis_sta_posum[j][:,:,None,None])/self.tp, 0) for j in range(3)]
            pos_loss_in = -torch.sum(torch.where(mask2[1], torch.log(self.eps+self.P(dis_new_pos[1], sta_ind[:,None], pos_ind[1], 'in')), torch.tensor(0.0, device=device)), dim=1)
            pos_loss_out = -torch.sum(torch.where(mask2[2], torch.log(self.eps+self.P(dis_new_pos[2], sta_ind[:,None], pos_ind[2], 'out')), torch.tensor(0.0, device=device)), dim=1)
            pos_loss = (pos_loss_in + pos_loss_out) / lg[0][:,None,None]
            dis_new_neg = torch.where(mask2[0], (dis_neg_cnc-dis_sta_neg[:,:,None,:]+dis_sta_negum[:,:,None,None])/self.tp, 1000)
            neg_loss = -torch.sum(torch.where(mask2[0], torch.log(self.eps+self.P(dis_new_neg, sta_ind[:,None], neg_ind, 'neg')), torch.tensor(0.0, device=device)), dim=1) / lg[0][:,None,None]
            total_loss = self.pos_ratio * pos_loss + neg_loss
            index = torch.argmin(total_loss, dim=1)
            i_indices, j_indices = torch.meshgrid(torch.arange(sta_ind.size(0)), torch.arange(self.tp), indexing='ij')
            self.locations[sta_ind[i_indices], j_indices] = cnc_loc[i_indices, index[i_indices, j_indices], j_indices]
            tl = torch.mean(total_loss[i_indices, index[i_indices, j_indices], j_indices])
            pl = torch.mean(pos_loss[i_indices, index[i_indices, j_indices], j_indices])
            nl = torch.mean(neg_loss[i_indices, index[i_indices, j_indices], j_indices])
        return tl, pl, nl

    def get_neighbor(self):
        # 获取边
        edges = torch.tensor(list(self.G.edges()), dtype=torch.long, device=device)
        src, dst = edges[:, 0], edges[:, 1]

        # 构建邻接矩阵
        idx_in  = torch.stack((dst, src), dim=0) # (2, E)
        idx_out = torch.stack((src, dst), dim=0) # (2, E)
        idx_all = torch.cat((idx_out, idx_in), dim=1) # (2, 2E)
        
        
        def get_sparse_adj(idx: torch.Tensor):
            val = torch.ones(idx.size(1), dtype=torch.bool, device=device)
            adj = torch.sparse_coo_tensor(
                indices=idx, 
                values =val,
                size=(self.num_nodes, self.num_nodes)
            ).to(device).coalesce()
            return adj

        self.adj = (get_sparse_adj(idx_all), get_sparse_adj(idx_in), get_sparse_adj(idx_out)) # (3, N, N), 稀疏

        # 获取 csr 索引：col 表示 0 ~ N-1 号节点的邻居排成一行，rowptr 表示每个节点的邻居起始位置，换句话说，对于节点 u, col[rowptr[u]:rowptr[u+1]] 就是 u 的所有邻居，或者 u 指出的边
        def coo_to_csr(adj: torch.Tensor):
            row, col = adj.indices() # 因为调用了 coalesce()，所以 row 和 col 都是有序的；也即，row 为第一关键字，col 为第二关键字，升序排列
            col = torch.cat((col, torch.tensor([-1], dtype=torch.int64, device=device)))
            counts = torch.bincount(row, minlength=adj.size(0)) # (N, )
            rowptr = torch.empty(self.num_nodes + 1, dtype=torch.int64, device=device)
            rowptr[0], rowptr[1:] = 0, torch.cumsum(counts, dim=0) # (N+1, )
            return rowptr, col
        
        self.rowptr, self.col = zip(*[coo_to_csr(adj) for adj in self.adj]) # (3, N+1), (3, E)
            
        # neighbor_all = [
        #     torch.sort(torch.tensor(list(set(self.G.predecessors(ii)) | set(self.G.successors(ii))), dtype=torch.int64, device=device))[0]
        #     for ii in range(self.num_nodes)
        # ]
        # neighbor_in = [
        #     torch.sort(torch.tensor(list(set(self.G.predecessors(ii))), dtype=torch.int64, device=device))[0]
        #     for ii in range(self.num_nodes)
        # ]
        # neighbor_out = [
        #     torch.sort(torch.tensor(list(set(self.G.successors(ii))), dtype=torch.int64, device=device))[0]
        #     for ii in range(self.num_nodes)
        # ]
        # neighbor = [neighbor_all, neighbor_in, neighbor_out]
        # neighbor_dict = [{ii: neighbor[j][ii] for ii in range(self.num_nodes)} for j in range(3)]
        # neighbor_tensor = [
        #     torch.full((self.num_nodes, self.max_degree), -1, dtype=torch.int64, device=device) 
        #     for _ in range(3)
        # ]
        # for j in range(3):
        #     for n, nbs in neighbor_dict[j].items():
        #         neighbor_tensor[j][n, :len(nbs)] = nbs
        # return neighbor_tensor
    
    
    def check_neighbor_consistency(self, neighbor_tensor, names=("all", "in", "out"), max_show=10):
        """
        比较两种构建方式得到的邻居是否完全一致：
        1) 稀疏邻接的 CSR 索引：self.rowptr[j], self.col[j]
        2) padding 矩阵：neighbor_tensor[j]，-1 代表 padding
        要求：
        - 调用前已执行过 self.get_neighbor()，并修正了 self.adj 的顺序与 coo_to_csr 的返回。
        - neighbor_tensor 与 self.rowptr/self.col 的类别顺序一致。
        参数：
        neighbor_tensor: List[Tensor]，长度为 3；每个形状为 (num_nodes, max_degree)
        names          : 每一类的名称，仅用于打印
        max_show       : 每类最多展示多少条不一致样例
        返回：
        ok             : bool，三类均完全一致返回 True，否则 False
        report         : dict，包含每类的 mismatch 计数与若干示例
        """
        assert hasattr(self, "rowptr") and hasattr(self, "col"), "call self.get_neighbor() first."

        ok = True
        report = {}

        for j in range(3):
            rowptr = self.rowptr[j].detach().cpu()
            col    = self.col[j].detach().cpu()
            nb_pad = neighbor_tensor[j].detach().cpu()

            mismatches = []
            total_nodes = int(rowptr.numel() - 1)

            for u in range(total_nodes):
                beg = int(rowptr[u].item())
                end = int(rowptr[u+1].item())
                # CSR 邻居集合
                csr_set = set(col[beg:end].tolist())
                # padding 邻居集合（过滤 -1）
                row = nb_pad[u]
                pad_set = set(row[row != -1].tolist())
                
                if csr_set != pad_set:
                    # 记录差异：仅展示差分的前若干个元素
                    only_in_csr = sorted(list(csr_set - pad_set))[:5]
                    only_in_pad = sorted(list(pad_set - csr_set))[:5]
                    mismatches.append({
                        "node": u,
                        "deg_csr": len(csr_set),
                        "deg_pad": len(pad_set),
                        "only_in_csr_head": only_in_csr,
                        "only_in_pad_head": only_in_pad,
                    })
                    if len(mismatches) >= max_show:
                        break

            report[names[j]] = {
                "num_nodes": total_nodes,
                "num_mismatched_nodes": len(mismatches),
                "examples": mismatches,
            }
            if len(mismatches) > 0:
                ok = False

        # 简要打印
        if not ok:
            print("=== Neighbor Consistency Check ===")
            for name in names:
                r = report[name]
                print(f"[{name}] nodes={r['num_nodes']}, mismatched={r['num_mismatched_nodes']}")
                for ex in r["examples"]:
                    print(f"  node={ex['node']}, deg_csr={ex['deg_csr']}, deg_pad={ex['deg_pad']}, "
                        f"only_in_csr_head={ex['only_in_csr_head']}, only_in_pad_head={ex['only_in_pad_head']}")
        
        return ok, report

    
    def neighbor_batch(self, sta_ind: torch.Tensor, epoch: int,
                       r1=None, r2=None):
        bs = sta_ind.size(0)
        batch_max_degree = self.degree[sta_ind].max().item()
        batch_neighbor = [self.neighbor_tensor[j][sta_ind, :batch_max_degree] for j in range(3)]
        batch_lengths = [self.degree[sta_ind], 
                        self.in_degree[sta_ind],
                        self.out_degree[sta_ind]]
        if epoch <= self.convergence * self.epoch:
            random_probs = torch.rand(bs, device=device) if r1 is None else r1 # (bs, )
            choosing_mask = random_probs > 0.2 # (bs, )
            batch_lengths = [torch.where(choosing_mask, batch_lengths[j], 1) for j in range(3)]
            
            one_random_neighbor = [((torch.rand(bs, device=device) if r2 is None else r2[j]) * batch_lengths[j]).floor().long() for j in range(3)] # (bs, )
            random_neighbor_mask = [torch.zeros((bs, batch_max_degree), dtype=torch.bool, device=device) for _ in range(3)] # (bs, max_degree)
            for j in range(3):
                random_neighbor_mask[j][torch.arange(bs), one_random_neighbor[j]] = True # (bs, max_degree)
                one_random_neighbor[j] = torch.full((bs, batch_max_degree), -1, dtype=torch.int64, device=device)
                one_random_neighbor[j][:, 0] = batch_neighbor[j][random_neighbor_mask[j]] # (bs, max_degree)         
                batch_neighbor[j] = torch.where(choosing_mask.unsqueeze(1), 
                                            batch_neighbor[j],
                                            one_random_neighbor[j]) # (bs, max_degree)
                # batch_neighbor[j] = batch_neighbor[j][:, :batch_lengths[j].max()]
        batch_mask = [batch_neighbor[j] != -1 for j in range(3)]
        return batch_neighbor, batch_lengths, batch_mask, batch_max_degree
    def debug_one_row(self, sta_ind, row_idx, pos_ind, mask, pos_ind_new, mask_new, j=2):
        # j: 0=all, 1=in, 2=out
        u = int(sta_ind[row_idx].item())
        print(f"[debug] category={('all','in','out')[j]}, batch_row={row_idx}, node_id={u}")

        # 旧版（padding 源）
        pad_row = self.neighbor_tensor[j][u].detach().cpu()
        pad_list = pad_row[pad_row != -1].tolist()
        print(f" old pad out-degree={len(pad_list)} head={pad_list[:12]}")

        # 新版（CSR 源）
        rp, col = self.rowptr[j].detach().cpu(), self.col[j].detach().cpu()
        beg, end = int(rp[u]), int(rp[u+1])
        csr_list = col[beg:end].tolist()
        print(f" new csr out-degree={len(csr_list)} head={csr_list[:12]}")

        # 两边集合是否一致
        s1, s2 = set(pad_list), set(csr_list)
        print(" set_equal(old,new)?", s1 == s2, " only_in_pad:", sorted((s1 - s2))[:12], " only_in_csr:", sorted((s2 - s1))[:12])

        # 本次 batch 两版 mask / neighbors 的头部
        print(" old mask true idx head:", mask[j][row_idx].nonzero(as_tuple=False).view(-1)[:16].tolist())
        print(" new mask true idx head:", mask_new[j][row_idx].nonzero(as_tuple=False).view(-1)[:16].tolist())
        print(" old pos_ind head:", pos_ind[j][row_idx, :16].tolist())
        print(" new pos_ind head:", pos_ind_new[j][row_idx, :16].tolist())

    def debug_check_neighbors(
        self,
        mode: str,           # "all" | "in" | "out"
        idx: int,
        sta_ind: torch.Tensor,  # (bs,)  int64
        nei: torch.Tensor,      # (bs,L) int64, -1 padding
        mask: torch.Tensor,     # (bs,L) bool
        num_nei: torch.Tensor   # (bs,)  int64
    ):
        """
        简单可读的合法性检查：
        1. mask.sum(dim=1) == num_nei
        2. 有效邻居无重复
        3. 所有有效邻居确实是对应邻接 (all/in/out) 的成员
        若发现问题则抛出 AssertionError。
        """
        assert mode in {"all", "in", "out"}, "mode 只能是 all|in|out"

        # 取对应 CSR
        rowptr = self.rowptr[idx]  # (N+1,) int64
        col    = self.col[idx]     # (E,)   int64

        bs, L = nei.shape
        for i in range(bs):
            u = int(sta_ind[i].item())

            # ---------- 1) mask 与 num_nei 一致 ----------
            masked_count = int(mask[i].sum().item())
            expected_count = int(num_nei[i].item())
            assert masked_count == expected_count, (
                f"[{mode}] 第 {i} 行 mask.count={masked_count} "
                f"但 num_nei={expected_count}"
            )

            # ---------- 2) 行内无重复 ----------
            valid_vals = nei[i][mask[i]].tolist()
            if len(valid_vals) != len(set(valid_vals)):
                raise AssertionError(
                    f"[{mode}] 第 {i} 行存在重复邻居：{valid_vals}"
                )

            # ---------- 3) 有效邻居确属该类别 ----------
            row_begin = int(rowptr[u].item())
            row_end   = int(rowptr[u + 1].item())
            legal_set = set(col[row_begin:row_end].tolist())

            illegal = [v for v in valid_vals if v not in legal_set]
            if illegal:
                raise AssertionError(
                    f"[{mode}] 第 {i} 行存在非法邻居 {illegal}，"
                    f"不在节点 {u} 的 {mode} 邻接中"
                )

        # print(f"✓ debug_check_neighbors({mode}) 通过，共 {bs} 行。")


    def neighbor_batch_fixed(self, sta_ind: torch.Tensor, epoch: int,
                             r1: Optional[torch.Tensor] = None,
                             r2: Optional[List[torch.Tensor]] = None,
                             fixed_L: Optional[int] = None):
        """
        返回:
        batch_neighbor = [nb_all, nb_in, nb_out]  # (bs, L), L=batch_max_degree
        batch_lengths  = [len_all, len_in, len_out]  # (bs,)
        batch_mask     = [mask_all, mask_in, mask_out]  # (bs, L)
        逻辑与原实现一致：L 取自 all-degree 的批内最大值；in/out 也按同一宽度 L 右侧用 -1 填充。
        """
        bs = sta_ind.size(0)
        deg_max: int = self.degree[sta_ind].max().item()  # type: ignore
        L: int = self.fixed_L if fixed_L is None else fixed_L # type: ignore
        L: int = min(L, deg_max)  # 确保 L 不超过批内最大度
        
        def neighbor_from_csr(rowptr: torch.Tensor,
                            col: torch.Tensor,
                            nodes: torch.Tensor,
                            L: int):
            """
            rowptr, col : CSR 索引 (GPU, int64)
            nodes       : (bs,)   GPU int64
            L           : int     每行返回的最大邻居数

            返回：
            nei   : (bs, L) int64，随机顺序，不足补 -1
            mask  : (bs, L) bool，True 表示有效
            num_n : (bs,)  int64，行内真实邻居数 (≤L)
            """
            device = rowptr.device
            bs = nodes.numel()

            # === 1. 行度 & 最大度 ===
            starts = rowptr[nodes]           # (bs,)
            ends   = rowptr[nodes + 1]       # (bs,)
            deg    = ends - starts           # (bs,)
            deg_max = int(deg.max().item())  # 标量，<= 全局 max_degree

            if deg_max == 0:
                # 批内全孤立节点
                nei  = torch.full((bs, L), -1, dtype=torch.long,  device=device)
                mask = torch.zeros((bs, L),      dtype=torch.bool, device=device)
                return nei, mask, deg            # num_nei 全 0

            # === 2. 为所有行构造 offset_matrix ∈ [0, deg_max) ===
            #    offsets[i, j] = j  (广播)
            offsets = torch.arange(deg_max, device=device).view(1, deg_max).expand(bs, -1)  # (bs, deg_max)

            # 有效位置掩码 valid[i, j] = (j < deg_i)
            valid = offsets < deg.unsqueeze(1)  # (bs, deg_max) bool

            # === 3. 行内随机乱序：对 valid 元素加随机噪声并 argsort ===
            noise = torch.empty_like(offsets, dtype=torch.float).uniform_()  # U(0,1)
            noise = torch.where(valid, noise, torch.full_like(noise, float('inf')))
            perm = noise.argsort(dim=1)                                      # (bs, deg_max), 行内由小→大随机顺序

            # 取前 L 列 → 行内无放回随机采样 L 个（若 deg_i<L 则后部是 invalid）
            perm_L = perm[:, :L]                                             # (bs, L)  offset indices
            mask   = perm_L < deg.unsqueeze(1)                               # (bs, L)  有效 True/False

            # === 4. 转成 col 索引并 gather ===
            # 对无效位置，随便给出合法 idx (如 starts)，随后再用 mask 覆盖为 -1
            safe_perm = torch.where(mask, perm_L, torch.zeros_like(perm_L))
            idx = starts.unsqueeze(1) + safe_perm                            # (bs, L)  col 索引
            nei_raw = col[idx]                                               # (bs, L)
            nei = torch.where(mask, nei_raw, torch.full_like(nei_raw, -1))

            num_nei = torch.minimum(deg, torch.as_tensor(L, device=device, dtype=deg.dtype))
            return nei, mask, num_nei

        nei, mask, num_nei = zip(*[
            neighbor_from_csr(self.rowptr[j], self.col[j], sta_ind, L)
            for j in range(3)
        ])
             
        if epoch <= self.convergence * self.epoch:
            random_probs = torch.rand(bs, device=device) if r1 is None else r1 # (bs, )
            choosing_mask = random_probs > 0.2 # (bs, )
            
            def keep_one_random(nei: torch.Tensor, num_nei: torch.Tensor, idx: int):
                one = torch.full_like(nei, -1) # (bs, L)
                num_nei_safe = torch.clamp(num_nei, min=1)  # (bs,)
                random_indices = ((torch.rand(bs, device=device) if r2 is None else r2[idx]) * num_nei_safe).floor().long()  # (bs,)
                random_indices = torch.clamp(random_indices, max=L-1).unsqueeze(1)  # (bs, 1)
                chosen_nei = torch.gather(nei, dim=1, index=random_indices)  # (bs, 1)
                chosen_nei = torch.where(
                    num_nei.unsqueeze(1) > 0,
                    chosen_nei,
                    -1
                )
                one[:, :1] = chosen_nei  # (bs, L), 仅第一个位置是随机选出的邻居
                nei_new = torch.where(
                    choosing_mask.unsqueeze(1),
                    nei,
                    one
                )  # (bs, L), 用 -1 填充
                num_nei_new = torch.where(
                    choosing_mask,
                    num_nei,
                    (chosen_nei != -1).long().squeeze()
                )  # (bs,), 至少为 1
                mask_new = nei_new >= 0 # (bs, L), 表示哪些位置是有效的邻居
                return nei_new, num_nei_new, mask_new
        
            nei, num_nei, mask = zip(*[
                keep_one_random(nei[j], num_nei[j], j) for j in range(3)
            ])
        

        return nei, num_nei, mask  
        
    def check_neighbor_batch_outputs_equal_setwise(
        self,
        pos_ind, lg, mask,
        pos_ind_new, lg_new, mask_new,
        names=("all", "in", "out"),
        max_show=5
    ):
        import torch

        def _ensure_triplet(x, name):
            if not isinstance(x, (list, tuple)) or len(x) != 3:
                raise ValueError(f"{name} 必须是长度为3的 list/tuple，对应 [all, in, out]")
            return x

        pos_ind     = _ensure_triplet(pos_ind,     "pos_ind")
        lg          = _ensure_triplet(lg,          "lg")
        mask        = _ensure_triplet(mask,        "mask")
        pos_ind_new = _ensure_triplet(pos_ind_new, "pos_ind_new")
        lg_new      = _ensure_triplet(lg_new,      "lg_new")
        mask_new    = _ensure_triplet(mask_new,    "mask_new")

        ok = True
        report = {
            "neighbors_set_mismatch": [],
            "lengths_mismatch": [],
            "masks_mismatch": []
        }

        for j in range(3):
            name = names[j]

            # 1) 先严格检查 mask 完全一致
            A_mask, B_mask = mask[j], mask_new[j]
            if A_mask.shape != B_mask.shape or A_mask.dtype != B_mask.dtype:
                ok = False
                report["masks_mismatch"].append({
                    "part": f"mask[{name}]",
                    "reason": f"shape/dtype mismatch: {tuple(A_mask.shape)}/{A_mask.dtype} vs {tuple(B_mask.shape)}/{B_mask.dtype}"
                })
                continue

            if not torch.equal(A_mask, B_mask):
                ok = False
                diff_rows = (~torch.all(A_mask == B_mask, dim=1)).nonzero(as_tuple=False).view(-1)
                examples = []
                for u in diff_rows[:max_show].tolist():
                    Au = A_mask[u].nonzero(as_tuple=False).view(-1)[:8].tolist()
                    Bu = B_mask[u].nonzero(as_tuple=False).view(-1)[:8].tolist()
                    examples.append({"row": int(u), "A_true_idx_head": Au, "B_true_idx_head": Bu})
                report["masks_mismatch"].append({
                    "part": f"mask[{name}]",
                    "reason": "content mismatch",
                    "num_mismatched_rows": int(diff_rows.numel()),
                    "examples": examples
                })
                continue  # mask 不一致时不再做该类的集合比较

            # 2) 检查 lg 完全一致
            A_lg, B_lg = lg[j], lg_new[j]
            if A_lg.shape != B_lg.shape or A_lg.dtype != B_lg.dtype:
                ok = False
                report["lengths_mismatch"].append({
                    "part": f"lengths[{name}]",
                    "reason": f"shape/dtype mismatch: {tuple(A_lg.shape)}/{A_lg.dtype} vs {tuple(B_lg.shape)}/{B_lg.dtype}"
                })
            elif not torch.equal(A_lg, B_lg):
                ok = False
                diff_idx = (A_lg != B_lg).nonzero(as_tuple=False).view(-1)
                examples = []
                for i in diff_idx[:max_show].tolist():
                    examples.append({"idx": int(i), "A_val": int(A_lg[i].item()), "B_val": int(B_lg[i].item())})
                report["lengths_mismatch"].append({
                    "part": f"lengths[{name}]",
                    "reason": "content mismatch",
                    "num_mismatched_indices": int(diff_idx.numel()),
                    "examples": examples
                })

            # 3) 比较 pos_ind 的有效邻居“集合”（顺序忽略；按行去重）
            A_nb, B_nb = pos_ind[j], pos_ind_new[j]
            if A_nb.shape != B_nb.shape or A_nb.dtype != B_nb.dtype:
                ok = False
                report["neighbors_set_mismatch"].append({
                    "part": f"neighbors[{name}]",
                    "reason": f"shape/dtype mismatch: {tuple(A_nb.shape)}/{A_nb.dtype} vs {tuple(B_nb.shape)}/{B_nb.dtype}"
                })
                continue

            mismatches = []
            bs, L = A_nb.shape
            A_nb_cpu = A_nb.detach().cpu()
            B_nb_cpu = B_nb.detach().cpu()
            Mask_cpu = A_mask.detach().cpu()  # 已知两者相等

            for u in range(bs):
                valid_idx = Mask_cpu[u].nonzero(as_tuple=False).view(-1)
                A_vals = A_nb_cpu[u, valid_idx].tolist()
                B_vals = B_nb_cpu[u, valid_idx].tolist()
                A_set = sorted(set(A_vals))
                B_set = sorted(set(B_vals))
                if A_set != B_set and len(mismatches) < max_show:
                    mismatches.append({
                        "row": int(u),
                        "A_set_head": A_set[:min(12, len(A_set))],
                        "B_set_head": B_set[:min(12, len(B_set))]
                    })

            if len(mismatches) > 0:
                ok = False
                report["neighbors_set_mismatch"].append({
                    "part": f"neighbors[{name}]",
                    "reason": "valid-neighbor sets differ (order-insensitive comparison)",
                    "num_mismatched_rows": len(mismatches),
                    "examples": mismatches
                })

        # === 修正后的安全打印 ===
        if not ok:
            print("=== neighbor_batch outputs setwise-equality check ===")
            print("NOT EQUAL: 发现不一致，详情如下：")
            for item in report.get("masks_mismatch", []):
                print(f"- {item['part']}: {item['reason']}")
                if "num_mismatched_rows" in item:
                    print(f"  num_mismatched_rows={item['num_mismatched_rows']}")
                    for ex in item.get("examples", []):
                        print(f"    row={ex['row']}, A_true_idx_head={ex['A_true_idx_head']}, B_true_idx_head={ex['B_true_idx_head']}")
            for item in report.get("lengths_mismatch", []):
                print(f"- {item['part']}: {item['reason']}")
                if "num_mismatched_indices" in item:
                    print(f"  num_mismatched_indices={item['num_mismatched_indices']}")
                    for ex in item.get("examples", []):
                        print(f"    idx={ex['idx']}, A_val={ex['A_val']}, B_val={ex['B_val']}")
            for item in report.get("neighbors_set_mismatch", []):
                print(f"- {item['part']}: {item['reason']}")
                if "num_mismatched_rows" in item:
                    print(f"  num_mismatched_rows={item['num_mismatched_rows']}")
                    for ex in item.get("examples", []):
                        print(f"    row={ex['row']}, A_set_head={ex['A_set_head']}, B_set_head={ex['B_set_head']}")

        return ok, report

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
        # self.degree = torch.IntTensor([self.G.degree(n) for n in self.G.nodes()]).cuda().to(torch.int64)
        self.degree = torch.IntTensor([
            len(set(self.G.predecessors(n)).union(self.G.successors(n)))
            for n in self.G.nodes()
        ]).cuda().to(torch.int64)
        
        self.max_degree = self.degree.max()
        # print("max_degree", self.max_degree)
        
        self.get_neighbor()
        # ok, _ = self.check_neighbor_consistency(self.neighbor_tensor, names=("all", "in", "out"))
        # if not ok:
        #     raise ValueError("Neighbor tensor is not consistent with CSR indices.")
        
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
        if MRR.get('evaluator') is not None:
            mrr_evaluator = MRR['evaluator']
            mrr1, mrr2 = mrr_evaluator.evaluate(pos, neg)
        else:
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
    parser.add_argument("--dataset", type=str, default='twitter')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--split_ratio", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=3)
    parser.add_argument("--h", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=3)
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--c", type=int, default=2)
    parser.add_argument("--neg", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--pos_ratio", type=float, default=1)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--convergence", type=float, default=0.8)
    parser.add_argument("--eval_step", type=int, default=10)
    parser.add_argument("--fixed_L", type=int, default=256, help="Fixed length for neighbor batch, if None will use max degree in the batch.")
    
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
    fixed_L = args.fixed_L
    direct_load = False
    
    set_random_seed(seed)
    print(f"dataset_name={dataset_name}, seed={seed}, epoch={epoch}, split_ratio={split_ratio}, alpha={alpha}, h={h}, gamma={gamma}, tp={tp}, c={c}, neg={neg}, batch_size={batch_size}, pos_ratio={pos_ratio}, chunks={chunks}, convergence={convergence}, eval_step={eval_step}, fixed_L={fixed_L}")
    
    # 5. Load the dataset
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
    if dataset_name == 'ogbl_citation2_node':
        MRR['evaluator'] = NodeMRREvaluator(train_data, test_data[0], test_data[1], use_fp16=True, max_memory=2.0)
    print("end to load", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # 6. Train and Test the model
    model = CritiGraph(h=h, tp=tp, c=c, eps=1e-5, neg=neg, gamma=gamma, 
                        alpha=alpha, epoch=epoch, batch_size=batch_size, pos_ratio=pos_ratio,
                        chunks=chunks, convergence=convergence, eval_step=eval_step, fixed_L=fixed_L)
    model(train_data, test_data[0], test_data[1])
    

    