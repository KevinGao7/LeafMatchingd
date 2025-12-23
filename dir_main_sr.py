import numpy as np
import os
import torch
import warnings
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from OurCogDL.cogdl.data import dataset
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
from torch import Tensor

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
    def __init__(self, h, tp, c, eps, neg, gamma, alpha, epoch, batch_edge, pos_ratio, 
                 chunks, convergence, eval_step, fixed_L, save_score):
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
        self.batch_edge = batch_edge
        self.pos_ratio = pos_ratio
        self.flip_masks = (1 << torch.arange(self.h, dtype=torch.int64, device=device)).unsqueeze(0).unsqueeze(2)
        self.distance_lookup_table = self.generate_distance_lookup_table()
        self.chunks = chunks
        self.convergence = convergence
        self.eval_step = eval_step
        self.fixed_L = fixed_L
        self.save_score = save_score
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
    
    def p_test(self, dis, ig1, ig2):
        deg1, deg2 = self.out_degree[ig1], self.in_degree[ig2]   
        ap = (deg1+1)*(deg2+1)
        aa = (dis+self.eps)/torch.log(ap)
        return 1/(1+aa**self.gamma/self.alpha)


    def _p_flat(self, dis: Tensor, ig1: Tensor, ig2: Tensor) -> Tensor:
        """
        扁平版本的 p，支持 dis 形状为 (E,L,tp) 或 (E,tp)。
        ig1, ig2: 形状 (E,) 的新图索引（已是 0..N-1）。
        返回: 与 dis 同形或可广播的概率张量。
        """
        deg1 = self.out_degree[ig1]  # (E,)
        deg2 = self.in_degree[ig2]   # (E,)
        ap = (deg1 + 1) * (deg2 + 1) # (E,)
        # 把 log(ap) reshape 成 (E,1,1) 以便对 (E,L,tp) 广播；如果 dis 是 (E,tp) 也能广播
        denom = torch.log(ap.float()).view(-1, *([1] * (dis.dim() - 1)))
        aa = (dis + self.eps) / denom
        return 1.0 / (1.0 + (aa**self.gamma) / self.alpha)

    def _P_flat(self, dis: Tensor, ig1: Tensor, ig2: Tensor, key: str) -> Tensor:
        """
        与原 P 完全等价的扁平版本：
        key='in'  -> p_21
        key='out' -> p_12
        key='neg' -> (1-p_12)*(1-p_21)
        """
        p_12 = self._p_flat(dis, ig1, ig2)
        p_21 = self._p_flat(dis, ig2, ig1)
        if key == 'in':
            return p_21
        elif key == 'out':
            return p_12
        elif key == 'neg':
            return (1.0 - p_12) * (1.0 - p_21)
        else:
            raise ValueError(f"Unknown key={key}")

    @torch.no_grad()
    def neighbor_batch_fixed_csr(self, sta_ind: torch.Tensor, choosing_mask_b: torch.Tensor):
        """
        基于三套 CSR（all/in/out）的纯张量邻居展开 + 20% 单邻抽样（对 choosing=False 的节点）
        输入:
        sta_ind         : (B,) int64，批内节点（0..N-1）
        choosing_mask_b : (B,) bool，True=保留全部邻居，False=仅保留 1 条（若该类邻接存在）
        依赖:
        self.rowptr, self.col : tuple 长度 3，对应 (all, in, out)，均为 int64 1-D
        其中 self.col[j] 的最后一个元素是 -1 哨兵，下游需用 col[:-1]
        输出:
        (src_all, dst_all, cnt_all, src_in, dst_in, src_out, dst_out)
        其中 src_* 是批内行号 [0..B-1]，可直接用于 index_add_。
        """
        # ---------- 基本断言 ----------
        assert sta_ind.dim() == 1 and sta_ind.dtype == torch.int64
        assert choosing_mask_b.dim() == 1 and choosing_mask_b.dtype == torch.bool
        B = sta_ind.numel()
        device = sta_ind.device

        assert isinstance(self.rowptr, (tuple, list)) and isinstance(self.col, (tuple, list))
        assert len(self.rowptr) == 3 and len(self.col) == 3
        ro_all, ro_in, ro_out = self.rowptr
        co_all, co_in, co_out = self.col

        # 形状/类型
        for ro, co in [(ro_all, co_all), (ro_in, co_in), (ro_out, co_out)]:
            assert ro.dim() == 1 and ro.dtype == torch.int64
            assert co.dim() == 1 and co.dtype == torch.int64
            assert ro.is_cuda and co.is_cuda

        N = ro_all.numel() - 1
        assert int(N) > 0
        assert sta_ind.min().item() >= 0 and sta_ind.max().item() < N
        assert choosing_mask_b.numel() == B

        # 去掉 col 最后一个哨兵 -1
        co_all = co_all[:-1]
        co_in  = co_in[:-1]
        co_out = co_out[:-1]

        # ---------- 一个小工具：从 CSR 展开扁平边，并按 choosing_mask_b 执行“单邻抽样” ----------
        def expand_and_sample(ro, co):
            """
            返回:
            src_flat, dst_flat, cnt_per_node_before_sample
            其中 cnt_per_node_before_sample 是 (B,) 的每个批内节点该类邻接的条数（抽样前）
            """
            off  = ro[sta_ind]                         # (B,)
            deg  = ro[sta_ind + 1] - off               # (B,)
            valid = deg > 0                             # (B,)

            # 扁平展开
            if valid.any():
                src = torch.repeat_interleave(torch.arange(B, device=device, dtype=torch.int64)[valid],
                                            deg[valid])                                  # (E,)
                base = off[valid].repeat_interleave(deg[valid])                             # (E,)
                lenv = deg[valid]
                head = torch.cumsum(lenv, dim=0) - lenv                                     # (Nv,)
                ofs  = torch.arange(lenv.sum().item(), device=device, dtype=torch.int64) - \
                    torch.repeat_interleave(head, lenv)                                   # (E,)
                dst = co[base + ofs]                                                         # (E,)
                # 统计每个批内节点的边数（抽样前）
                bc = torch.bincount(src, minlength=B)                                        # (B,)
                # 对 choosing=False 的节点，仅保留 1 条
                if (~choosing_mask_b[src]).any():
                    keep_all = choosing_mask_b[src]                                          # (E,)
                    head_e = torch.cumsum(bc, 0) - bc                                        # (B,)
                    idx_in_seg = torch.arange(src.numel(), device=device, dtype=torch.int64) - head_e[src]
                    seg_len = bc.clamp_min(1)                                                # (B,)
                    pick = (torch.rand(B, device=device) * seg_len.float()).floor().to(torch.int64)
                    keep = keep_all | (idx_in_seg == pick[src])
                    src, dst = src[keep], dst[keep]
            else:
                # 该类邻接在本批内为空
                src = torch.empty(0, dtype=torch.int64, device=device)
                dst = torch.empty(0, dtype=torch.int64, device=device)
                bc  = torch.zeros(B, dtype=torch.int64, device=device)

            return src, dst, bc

        # in / out 展开与抽样
        src_in,  dst_in,  cnt_in_before  = expand_and_sample(ro_in,  co_in)
        src_out, dst_out, cnt_out_before = expand_and_sample(ro_out, co_out)

        # all 展开与抽样（注意：all 已经是 in ∪ out 的去重并集，你的构造保证这一点）
        src_all, dst_all, cnt_all_before = expand_and_sample(ro_all, co_all)

        # ---------- 计算 cnt_all（归一化分母）：抽样后计数 ----------
        # 对 choosing=True 的节点：cnt_all = 抽样前计数；对 choosing=False 且该类存在：cnt_all = 1；否则 0
        cnt_all = torch.where(
            choosing_mask_b,
            cnt_all_before,
            (cnt_all_before > 0).to(torch.int64)
        )
        # 分母保护，与原逻辑一致
        cnt_all = cnt_all.clamp_min(1)

        # ---------- 最小一致性断言 ----------
        assert src_in.numel() == dst_in.numel()
        assert src_out.numel() == dst_out.numel()
        assert src_all.numel() == dst_all.numel()
        assert cnt_all.shape == (B,) and cnt_all.dtype == torch.int64

        return (src_all, dst_all, cnt_all,
                src_in,  dst_in,
                src_out, dst_out)



    # 在 CritiGraph 类内，替换 loom_v2 签名，并把 _neighbor_batch_flat 调用改为传入 choosing_mask_b
    @torch.no_grad()
    def loom_v2(self, epoch: int, sta_ind: torch.Tensor, choosing_mask_b: torch.Tensor):
        device = sta_ind.device
        B = sta_ind.size(0)
        tp = self.tp

        sta_loc = self.locations[sta_ind]           # (B,tp)
        cnc_loc = self.connection(sta_loc)          # (B,L,tp)
        L = cnc_loc.size(1)
        perm = torch.randperm(L, device=device)
        cnc_loc = cnc_loc[:, perm, :]

        (src_all, dst_all, cnt_all,
        src_in,  dst_in,
        src_out, dst_out) = self.neighbor_batch_fixed_csr(sta_ind, choosing_mask_b)


        # 负采样：与 all 对齐
        E_all = src_all.numel()
        neg_dst_all = torch.randint(0, self.num_nodes, (E_all,), device=device, dtype=torch.long)

        def node_locs(idx_1d): return self.locations[idx_1d]
        def gather_src(src_flat):
            return sta_loc[src_flat], cnc_loc[src_flat]  # (E,tp), (E,L,tp)
        eps = self.eps

        def pos_branch(src_flat, dst_flat, key: str):
            E = src_flat.numel()
            if E == 0:
                return torch.zeros((B, L, tp), device=device)
            sta_src_vec, cnc_src_vec = gather_src(src_flat)
            pos_dst_vec = node_locs(dst_flat)
            dis_sta_pos = self.distance(sta_src_vec, pos_dst_vec).float()        # (E,tp)
            dis_sta_posum = dis_sta_pos.sum(dim=-1)                               # (E,)
            dis_pos_cnc = self.distance(cnc_src_vec, pos_dst_vec.unsqueeze(1)).float()  # (E,L,tp)
            dis_new_pos = (dis_pos_cnc - dis_sta_pos.unsqueeze(1) + dis_sta_posum.view(-1,1,1)) / tp
            P_edge = self._P_flat(dis_new_pos, sta_ind[src_flat], dst_flat, key)  # (E,L,tp)
            ll = -torch.log(eps + P_edge)
            acc = torch.zeros((B, L, tp), device=device)
            acc.index_add_(0, src_flat, ll)
            return acc

        pos_loss_in  = pos_branch(src_in,  dst_in,  'in')
        pos_loss_out = pos_branch(src_out, dst_out, 'out')
        pos_loss = (pos_loss_in + pos_loss_out) / cnt_all.view(-1,1,1)

        if E_all == 0:
            neg_loss = torch.zeros((B, L, tp), device=device)
        else:
            sta_src_vec, cnc_src_vec = gather_src(src_all)
            neg_dst_vec = node_locs(neg_dst_all)
            dis_sta_neg = self.distance(sta_src_vec, neg_dst_vec).float()        # (E_all,tp)
            dis_sta_negum = dis_sta_neg.sum(dim=-1)                               # (E_all,)
            dis_neg_cnc = self.distance(cnc_src_vec, neg_dst_vec.unsqueeze(1)).float()  # (E_all,L,tp)
            dis_new_neg = (dis_neg_cnc - dis_sta_neg.unsqueeze(1) + dis_sta_negum.view(-1,1,1)) / tp
            P_neg = self._P_flat(dis_new_neg, sta_ind[src_all], neg_dst_all, 'neg')
            ll_neg = -torch.log(eps + P_neg)
            neg_acc = torch.zeros((B, L, tp), device=device)
            neg_acc.index_add_(0, src_all, ll_neg)
            neg_loss = neg_acc / cnt_all.view(-1,1,1)

        total_loss = self.pos_ratio * pos_loss + neg_loss
        index = torch.argmin(total_loss, dim=1)  # (B,tp)

        b_idx = torch.arange(B, device=device)[:, None]
        t_idx = torch.arange(tp, device=device)[None, :]
        self.locations[sta_ind[b_idx], t_idx] = cnc_loc[b_idx, index, t_idx]

        def gather_bt(M): return M[b_idx, index, t_idx].mean()
        tl = gather_bt(total_loss); pl = gather_bt(pos_loss); nl = gather_bt(neg_loss)
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
    
    
    
    def edge_capped_node_batches(self, epoch: int, num_edges_cap: int, shuffle: bool = True):
        """
        依照 num_edges_cap 生成一批批节点，使得每个 batch 的“有效边数”
        E_batch := (all 有效边) + (in 有效边) + (out 有效边) <= num_edges_cap。
        有效边定义与训练采样一致：在收敛前期，20% 节点每类邻居仅保留 1 条（若原本>0）。
        产出: (sta_ind_b, choosing_mask_b)  二者均在 GPU 上。
        """
        device = self.out_degree.device

        # 1) 本 epoch 的全局 choosing_mask（逐节点），与训练采样一致
        if epoch <= self.convergence * self.epoch:
            choosing_mask_global = (torch.rand(self.num_nodes, device=device) > 0.2)
        else:
            choosing_mask_global = torch.ones(self.num_nodes, dtype=torch.bool, device=device)

        # 只考虑度>0的节点（与训练一致）
        nodes = self.li.clone()
        if shuffle:
            perm = torch.randperm(nodes.numel(), device=device)
            nodes = nodes[perm]

        # 2) 计算每个候选节点在本 epoch 下的“有效边数”
        deg_all = self.degree[nodes]
        deg_in  = self.in_degree[nodes]
        deg_out = self.out_degree[nodes]
        mask_b  = choosing_mask_global[nodes]

        eff_all = torch.where(mask_b, deg_all, deg_all.clamp_max(1))
        eff_in  = torch.where(mask_b, deg_in,  deg_in.clamp_max(1))
        eff_out = torch.where(mask_b, deg_out, deg_out.clamp_max(1))
        cost    = eff_all + eff_in + eff_out   # 每个节点的有效边数（与真正构边一致）

        # 3) 顺序装箱，边数超过上限前切 batch
        cur_nodes = []
        cur_mask  = []
        cur_edges = 0

        # 注意：控制流放在 CPU 可降低 GPU 同步开销
        cost_cpu = cost.detach().cpu().tolist()
        nodes_cpu = nodes.detach().cpu().tolist()
        mask_cpu  = mask_b.detach().cpu().tolist()

        for n_id, c, m in zip(nodes_cpu, cost_cpu, mask_cpu):
            c = int(c)
            if c == 0:
                # 该节点虽在 li 中，但三类邻居在本 epoch 的采样后都为空（极少见），跳过
                continue
            # 若当前非空且加入后超过上限，则先发出当前 batch
            if len(cur_nodes) > 0 and (cur_edges + c) > num_edges_cap:
                sta_ind_b = torch.tensor(cur_nodes, device=device, dtype=torch.long)
                choosing_mask_b = torch.tensor(cur_mask, device=device, dtype=torch.bool)
                yield sta_ind_b, choosing_mask_b
                cur_nodes, cur_mask, cur_edges = [], [], 0

            # 把当前节点放入 batch；若它本身 c > num_edges_cap，将成为“重节点”单批
            cur_nodes.append(n_id)
            cur_mask.append(bool(m))
            cur_edges += c

            # 如果单节点已超过上限，立即发出（避免死循环）
            if cur_edges > num_edges_cap:
                sta_ind_b = torch.tensor(cur_nodes, device=device, dtype=torch.long)
                choosing_mask_b = torch.tensor(cur_mask, device=device, dtype=torch.bool)
                yield sta_ind_b, choosing_mask_b
                cur_nodes, cur_mask, cur_edges = [], [], 0

        if len(cur_nodes) > 0:
            sta_ind_b = torch.tensor(cur_nodes, device=device, dtype=torch.long)
            choosing_mask_b = torch.tensor(cur_mask, device=device, dtype=torch.bool)
            yield sta_ind_b, choosing_mask_b
         
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
        
        if MRR_NODE.run:
            MRR_NODE.init(self.num_nodes, block_size=16384, neg_ratio=1000)
        
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
        
        for epoch in range(self.epoch):
            print("current_time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "pos+neg, pos, neg")
            sys.stdout.flush()

            # 将标量改为浮点累计
            total_wsum = 0.0
            pos_wsum   = 0.0
            neg_wsum   = 0.0
            node_count = 0   # 本 epoch 参与训练的“节点总数”（度>0）
            total_time = 0.0

            # —— 按边数限额的分批 —— 
            for sta_ind_b, choosing_mask_b in self.edge_capped_node_batches(epoch, self.batch_edge, shuffle=True):
                # print("epoch", epoch, "processing batch with", sta_ind_b.numel(), "nodes")
                st_time = datetime.now()
                tl, pl, nl = self.loom_v2(epoch, sta_ind_b, choosing_mask_b)  # tl/pl/nl: 已是(按 b,t)平均的 batch 标量
                B = sta_ind_b.numel()

                # 按节点数加权累加
                total_wsum += tl.item() * B
                pos_wsum   += pl.item() * B
                neg_wsum   += nl.item() * B
                node_count += B
                ed_time = datetime.now()
                total_time += (ed_time - st_time).total_seconds()

            print("Finished training epoch:", epoch, "total time:", total_time)

            # —— 用节点总数做归一化（恢复原始“按节点均值”的语义） ——
            total    = total_wsum / max(1, node_count)
            positive = pos_wsum   / max(1, node_count)
            negative = neg_wsum   / max(1, node_count)

            print(epoch, float(total), float(positive), float(negative))

            if (epoch + 1) % self.eval_step == 0 or epoch == self.epoch - 1:
                print("current_time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print("start to eval"); sys.stdout.flush()
                self.eval(test_pos, test_neg, epoch)
    
    def get_score(self, data): # shape = (E, 2)
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
        # 输出 10%, 20%, ..., 90%, 100% 的正 / 负样本分数分位点
        # for p in np.arange(90, 100, 1):
        #     pos_perc = torch.quantile(pos, p / 100.0).item()
        #     neg_perc = torch.quantile(neg, p / 100.0).item()
        #     print(f"Percentile {p}%: pos score = {pos_perc}, neg score = {neg_perc}")
            
        # for p in np.arange(99., 100.1, 0.1):
        #     pos_perc = torch.quantile(pos, p / 100.0).item()
        #     neg_perc = torch.quantile(neg, p / 100.0).item()
        #     print(f"Percentile {p}%: pos score = {pos_perc}, neg score = {neg_perc}")
            
        if self.save_score is not None:
            pos_np = pos.detach().cpu().numpy()
            neg_np = neg.detach().cpu().numpy()
            if not os.path.exists(self.save_score):
                os.makedirs(self.save_score)
            np.save(f"{self.save_score}/pos.npy", pos_np)
            np.save(f"{self.save_score}/neg.npy", neg_np)

        sorted_pos = torch.sort(pos)[0]
        sorted_neg = torch.sort(neg)[0]
        # neg_len = sorted_neg.size(0)
        # sorted_neg = sorted_neg[:int(neg_len * 0.99)]
        
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
        print('roc_auc, pr_auc, f1, mrr_pess, mrr_opt', roc_auc, pr_auc, f1, mrr_pess, mrr_opt)            

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
    parser.add_argument("--batch_edge", type=int, default=128)
    parser.add_argument("--pos_ratio", type=float, default=1)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--convergence", type=float, default=0.8)
    parser.add_argument("--eval_step", type=int, default=10)
    parser.add_argument("--save_score", type=str, default=None)
    parser.add_argument("--save_emb", type=str, default=None)
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
    batch_edge = args.batch_edge
    pos_ratio = args.pos_ratio
    chunks = args.chunks
    convergence = args.convergence
    eval_step = args.eval_step
    fixed_L = args.fixed_L
    direct_load = False
    save_score = args.save_score
    save_emb = args.save_emb
    
    set_random_seed(seed)
    print(f"dataset_name={dataset_name}, seed={seed}, epoch={epoch}, split_ratio={split_ratio}, alpha={alpha}, h={h}, gamma={gamma}, tp={tp}, c={c}, neg={neg}, batch_edge={batch_edge}, pos_ratio={pos_ratio}, chunks={chunks}, convergence={convergence}, eval_step={eval_step}, fixed_L={fixed_L}")
    
    # 5. Load the dataset
    if dataset_name.startswith("ER"):
        data_path = f'./datasets/{dataset_name}/data.pt'  
        direct_load = True 
    elif dataset_name == 'ogbl_citation2':
        MRR_NODE.activate()
        data_path = f'./datasets/{dataset_name}/split/'
    else:
        data_path = f'./datasets/{dataset_name}/split/'
        
    print("start to load", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    train_data = get_train_pt(data_path, split_ratio, 
                            direct_load=direct_load, dataset_name=dataset_name)
    test_data = get_test_pt(data_path, split_ratio, 
                            direct_load=direct_load, dataset_name=dataset_name)
    print("end to load", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # 6. Train and Test the model
    model = CritiGraph(h=h, tp=tp, c=c, eps=1e-5, neg=neg, gamma=gamma, 
                        alpha=alpha, epoch=epoch, batch_edge=batch_edge, pos_ratio=pos_ratio,
                        chunks=chunks, convergence=convergence, eval_step=eval_step, fixed_L=fixed_L,
                        save_score=save_score)
    model(train_data, test_data[0], test_data[1])

    if save_emb:
        torch.save(
            {
                'locations': model.locations.cpu(),
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

    