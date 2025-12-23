import numpy as np
import torch
import warnings
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import networkx as nx
warnings.filterwarnings('ignore', 'divide by zero encountered in log2')
import torch
import argparse
import random
import sys

# setting global device ...
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# defining dataset ...
class MyDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        sta_ind = self.indices[idx]
        return sta_ind.clone().detach().to(dtype=torch.int64).cuda() 

def get_train(name, ratio):
    edges = []
    file_path = './ori_datasets/'+name+'/split_'+ratio+'/train_pos.txt'
    with open(file_path, 'r') as file:
        for line in file:
            edge = line.strip().split()
            edges.append([int(edge[0]), int(edge[1])])
    return torch.tensor(edges, dtype=torch.int64)

def get_test(name, ratio):
    edges_p, edges_n = [], []
    file_path_p, file_path_n = './ori_datasets/'+name+'/split_'+ratio+'/test_pos.txt', './ori_datasets/'+name+'/split_'+ratio+'/test_neg.txt'
    with open(file_path_p, 'r') as file:
        for line in file:
            edge = line.strip().split()
            edges_p.append([int(edge[0]), int(edge[1])])
    with open(file_path_n, 'r') as file:
        for line in file:
            edge = line.strip().split()
            edges_n.append([int(edge[0]), int(edge[1])])
    return torch.tensor(edges_p, dtype=torch.int64), torch.tensor(edges_n, dtype=torch.int64)


# defining model ...
class CritiGraph(torch.nn.Module):
    def __init__(self, h, tp, c, eps, neg, gamma, lam, alpha, epoch, batch_size, save_score=None):
        super().__init__() 
        self.h = h
        self.tp = tp
        self.n = int(2**h)
        self.c = c
        self.k = int(c*h)
        self.eps = eps
        self.neg = neg
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha
        self.epoch = epoch  

        # 原 batch_size 作为“每个 batch 的有效边数上限”
        self.batch_size = batch_size
        self.batch_edge = batch_size

        self.flip_masks = (1 << torch.arange(self.h, dtype=torch.int64, device=device)).unsqueeze(0).unsqueeze(2)
        self.distance_lookup_table = self.generate_distance_lookup_table()
        self.save_score = save_score

    def __call__(self, graph):
        return self.forward(graph)

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
        random_numbers = torch.randint(0, self.n, (self.h, sz, self.k, self.tp), dtype=torch.int64, device=device)
        masks = random_numbers % upper_bounds.view(-1, 1, 1, 1)
        return masks.permute(1, 0, 2, 3)

    def connection(self, ori_int):
        flipped_ints = ori_int.unsqueeze(1) ^ self.flip_masks              # (B,h,tp)
        random_masks = self.generate_random_masks(flipped_ints.size(0))     # (B,h,k,tp)
        result = (flipped_ints.unsqueeze(2) ^ random_masks).view(flipped_ints.size(0), self.h*self.k, self.tp)
        return torch.cat((result, ori_int.unsqueeze(1)), dim=1)             # (B,h*k+1,tp)

    def p(self, dis, ig1, ig2):
        deg1, deg2 = self.degree[ig1], self.degree[ig2]        
        ap = (deg1 + 1) * (deg2 + 1)
        aa = (dis + self.eps) / torch.log(ap)[:,:,None,None]
        return 1 / (1 + aa ** self.gamma / self.alpha)

    def p_test(self, dis, ig1, ig2):
        deg1, deg2 = self.degree[ig1], self.degree[ig2]        
        ap = (deg1 + 1) * (deg2 + 1)
        aa = (dis + self.eps)/torch.log(ap)
        return 1 / (1 + aa ** self.gamma/self.alpha)

    def _p_flat(self, dis: torch.Tensor, ig1: torch.Tensor, ig2: torch.Tensor) -> torch.Tensor:
        """
        扁平版本的 p，支持 dis 形状为 (E,L,tp) 或 (E,tp)。
        ig1, ig2: (E,) 新图索引。
        """
        deg1 = self.degree[ig1]   # (E,)
        deg2 = self.degree[ig2]   # (E,)
        ap = (deg1 + 1) * (deg2 + 1)  # (E,)
        denom = torch.log(ap.float()).view(-1, *([1] * (dis.dim() - 1)))
        aa = (dis + self.eps) / denom
        return 1.0 / (1.0 + (aa ** self.gamma) / self.alpha)

    def get_neighbor(self):
        """
        基于无向图 self.G 构造一套 CSR 邻接 (rowptr, col)。
        对每条无向边 {u,v}，在邻接中加入 (u,v) 和 (v,u)。
        """
        edges = torch.tensor(list(self.G.edges()), dtype=torch.long, device=device)
        if edges.numel() == 0:
            self.rowptr = torch.zeros(self.num_nodes + 1, dtype=torch.long, device=device)
            self.col = torch.tensor([-1], dtype=torch.long, device=device)
            return

        src = edges[:, 0]
        dst = edges[:, 1]

        # 无向 -> 两条有向边
        idx = torch.cat(
            (
                torch.stack((src, dst), dim=0),
                torch.stack((dst, src), dim=0),
            ),
            dim=1
        )  # 2 x (2E)

        val = torch.ones(idx.size(1), dtype=torch.bool, device=device)
        adj = torch.sparse_coo_tensor(
            indices=idx,
            values=val,
            size=(self.num_nodes, self.num_nodes)
        ).coalesce()

        row, col = adj.indices()  # row 升序
        col = torch.cat((col, torch.tensor([-1], dtype=torch.long, device=device)))
        counts = torch.bincount(row, minlength=self.num_nodes)
        rowptr = torch.empty(self.num_nodes + 1, dtype=torch.long, device=device)
        rowptr[0] = 0
        rowptr[1:] = torch.cumsum(counts, dim=0)

        self.rowptr = rowptr
        self.col = col

    @torch.no_grad()
    def neighbor_batch_csr(self, sta_ind: torch.Tensor, choosing_mask_b: torch.Tensor):
        """
        基于 CSR 邻接 (rowptr, col) 展开 batch 内所有无向边，
        并对 choosing_mask_b=False 的节点执行“只保留 1 条邻居”的抽样。

        输入:
            sta_ind        : (B,) int64，batch 内节点索引
            choosing_mask_b: (B,) bool
        返回:
            src_flat, dst_flat, cnt_all
        """
        assert sta_ind.dim() == 1 and sta_ind.dtype == torch.int64
        assert choosing_mask_b.dim() == 1 and choosing_mask_b.dtype == torch.bool
        B = sta_ind.numel()
        device_local = sta_ind.device

        ro = self.rowptr                      # (N+1,)
        co = self.col[:-1]                    # 去掉哨兵

        off = ro[sta_ind]                     # (B,)
        deg = ro[sta_ind + 1] - off           # (B,)
        valid = deg > 0                       # (B,)

        if valid.any():
            sta_ids_valid = torch.arange(B, device=device_local, dtype=torch.long)[valid]
            lenv = deg[valid]                                    # (B_valid,)
            src_flat = torch.repeat_interleave(sta_ids_valid, lenv)   # (E,)

            base = off[valid].repeat_interleave(lenv)                 # (E,)

            head = torch.cumsum(lenv, dim=0) - lenv                   # (B_valid,)
            ofs = torch.arange(lenv.sum().item(), device=device_local, dtype=torch.long) - \
                  torch.repeat_interleave(head, lenv)                 # (E,)

            dst_flat = co[base + ofs]                                 # (E,)
            bc = torch.bincount(src_flat, minlength=B)                # (B,)

            if (~choosing_mask_b[src_flat]).any():
                keep_all = choosing_mask_b[src_flat]                  # (E,)
                head_e = torch.cumsum(bc, 0) - bc                     # (B,)
                idx_in_seg = torch.arange(src_flat.numel(), device=device_local, dtype=torch.long) - head_e[src_flat]
                seg_len = bc.clamp_min(1)
                pick = (torch.rand(B, device=device_local) * seg_len.float()).floor().to(torch.long)
                keep = keep_all | (idx_in_seg == pick[src_flat])
                src_flat = src_flat[keep]
                dst_flat = dst_flat[keep]
                bc = torch.bincount(src_flat, minlength=B)
        else:
            src_flat = torch.empty(0, dtype=torch.long, device=device_local)
            dst_flat = torch.empty(0, dtype=torch.long, device=device_local)
            bc = torch.zeros(B, dtype=torch.long, device=device_local)

        cnt_all = torch.where(
            choosing_mask_b,
            bc,
            (bc > 0).to(torch.long)
        )
        cnt_all = cnt_all.clamp_min(1)

        return src_flat, dst_flat, cnt_all

    def edge_capped_node_batches(self, epoch: int, num_edges_cap: int, shuffle: bool = True):
        """
        按“有效边数上限 num_edges_cap”生成一批批节点：
            对每个 batch，有效边数 E_batch <= num_edges_cap。

        有效边定义：
            epoch <= 4/5 * self.epoch 时：
                ~20% 节点只保留 1 条邻居，其余节点保留全部邻居；
            之后：
                所有节点保留全部邻居。
        产出:
            (sta_ind_b, choosing_mask_b) 均在 GPU。
        """
        device_local = self.degree.device

        if epoch <= 4 * self.epoch / 5.0:
            choosing_mask_global = (torch.rand(self.num_nodes, device=device_local) > 0.2)
        else:
            choosing_mask_global = torch.ones(self.num_nodes, dtype=torch.bool, device=device_local)

        nodes = self.li.clone()
        if shuffle:
            perm = torch.randperm(nodes.numel(), device=device_local)
            nodes = nodes[perm]

        deg_all = self.degree[nodes]              # (M,)
        mask_b = choosing_mask_global[nodes]      # (M,)

        eff_all = torch.where(mask_b, deg_all, deg_all.clamp_max(1))
        cost = eff_all

        cur_nodes = []
        cur_mask = []
        cur_edges = 0

        cost_cpu = cost.detach().cpu().tolist()
        nodes_cpu = nodes.detach().cpu().tolist()
        mask_cpu = mask_b.detach().cpu().tolist()

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
    def loom(self, epoch: int, sta_ind: torch.Tensor, choosing_mask_b: torch.Tensor):
        """
        基于 CSR 邻接和“以边为单位”的采样逻辑的训练步。
        输入:
            sta_ind        : (B,) 节点索引
            choosing_mask_b: (B,) bool
        返回:
            tl, pl, nl: 标量，总 loss / 正 loss / 负 loss
        """
        device_local = sta_ind.device
        B = sta_ind.size(0)
        tp = self.tp

        sta_loc = self.locations[sta_ind]           # (B,tp)
        cnc_loc = self.connection(sta_loc)          # (B,L,tp)
        L = cnc_loc.size(1)
        perm = torch.randperm(L, device=device_local)
        cnc_loc = cnc_loc[:, perm, :]

        src_flat, dst_flat, cnt_all = self.neighbor_batch_csr(sta_ind, choosing_mask_b)
        E = src_flat.numel()
        if E == 0:
            zero = torch.tensor(0.0, device=device_local)
            return zero, zero, zero

        neg_dst_flat = torch.randint(
            0, self.num_nodes, (E,),
            dtype=torch.long, device=device_local
        )

        # 正样本
        sta_src_vec = sta_loc[src_flat]                # (E,tp)
        cnc_src_vec = cnc_loc[src_flat]                # (E,L,tp)
        pos_dst_vec = self.locations[dst_flat]         # (E,tp)

        dis_sta_pos = self.distance(sta_src_vec, pos_dst_vec).float()          # (E,tp)
        dis_sta_posum = dis_sta_pos.sum(dim=-1)                                # (E,)
        dis_pos_cnc = self.distance(cnc_src_vec, pos_dst_vec.unsqueeze(1)).float()  # (E,L,tp)
        dis_new_pos = (dis_pos_cnc
                       - dis_sta_pos.unsqueeze(1)
                       + dis_sta_posum.view(-1, 1, 1)) / tp                    # (E,L,tp)

        ig1_pos = sta_ind[src_flat]                # (E,)
        ig2_pos = dst_flat                         # (E,)
        P_pos = self._p_flat(dis_new_pos, ig1_pos, ig2_pos)                    # (E,L,tp)
        ll_pos = -torch.log(self.eps + P_pos)                                  # (E,L,tp)

        pos_acc = torch.zeros((B, L, tp), dtype=torch.float32, device=device_local)
        pos_acc.index_add_(0, src_flat, ll_pos)                                # (B,L,tp)
        pos_loss = pos_acc / cnt_all.view(-1, 1, 1)                            # (B,L,tp)

        # 负样本
        neg_dst_vec = self.locations[neg_dst_flat]                             # (E,tp)
        dis_sta_neg = self.distance(sta_src_vec, neg_dst_vec).float()          # (E,tp)
        dis_sta_negum = dis_sta_neg.sum(dim=-1)                                # (E,)
        dis_neg_cnc = self.distance(cnc_src_vec, neg_dst_vec.unsqueeze(1)).float()  # (E,L,tp)
        dis_new_neg = (dis_neg_cnc
                       - dis_sta_neg.unsqueeze(1)
                       + dis_sta_negum.view(-1, 1, 1)) / tp                    # (E,L,tp)

        ig1_neg = ig1_pos
        ig2_neg = neg_dst_flat
        P_neg = self._p_flat(dis_new_neg, ig1_neg, ig2_neg)                    # (E,L,tp)
        ll_neg = -torch.log(1.0 + self.eps - P_neg)                            # (E,L,tp)

        neg_acc = torch.zeros((B, L, tp), dtype=torch.float32, device=device_local)
        neg_acc.index_add_(0, src_flat, ll_neg)                                # (B,L,tp)
        neg_loss = neg_acc / cnt_all.view(-1, 1, 1)                            # (B,L,tp)

        total_loss = self.lam * pos_loss + neg_loss                            # (B,L,tp)
        index = torch.argmin(total_loss, dim=1)                                # (B,tp)

        b_idx = torch.arange(B, device=device_local)[:, None]                  # (B,1)
        t_idx = torch.arange(tp, device=device_local)[None, :]                 # (1,tp)
        self.locations[sta_ind[b_idx], t_idx] = cnc_loc[b_idx, index, t_idx]

        gather_bt = lambda M: M[b_idx, index, t_idx].mean()
        tl = gather_bt(total_loss)
        pl = gather_bt(pos_loss)
        nl = gather_bt(neg_loss)
        return tl, pl, nl

    def forward(self, graph):
        current_time = datetime.now()
        print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
        print("start to load data")
        self.Go = nx.Graph()
        for edge in graph:
            self.Go.add_edge(edge[0].item(), edge[1].item())
        
        old_nodes = list(self.Go.nodes())
        self.mapping = {old_node: new_node for new_node, old_node in enumerate(old_nodes)}
        self.G = nx.relabel_nodes(self.Go, self.mapping)

        self.num_nodes = self.G.number_of_nodes()
        self.num_edges = self.G.number_of_edges()
        print('num_nodes', self.num_nodes)
        print('num_edges', self.num_edges)
        
        print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
        self.degree = torch.IntTensor([self.G.degree(n) for n in self.G.nodes()]).cuda().to(torch.int64)
        self.max_degree = self.degree.max()

        # 构建 CSR 邻接
        self.get_neighbor()

        # 初始化 embedding 与度 > 0节点集合
        self.locations = torch.randint(0, self.n, (self.num_nodes, self.tp), dtype=torch.int64, device=device)
        self.li = torch.arange(self.num_nodes, dtype=torch.int64, device=device)[self.degree > 0]

        for epoch in range(self.epoch):
            sys.stdout.flush()
            print("current_time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "pos+neg, pos, neg")

            total_wsum = 0.0
            pos_wsum = 0.0
            neg_wsum = 0.0
            node_count = 0
            total_time = 0.0

            for sta_ind_b, choosing_mask_b in self.edge_capped_node_batches(epoch, self.batch_edge, shuffle=True):
                
                st_time = datetime.now()
                tl, pl, nl = self.loom(epoch, sta_ind_b, choosing_mask_b)
                B = sta_ind_b.numel()

                total_wsum += tl.item() * B
                pos_wsum += pl.item() * B
                neg_wsum += nl.item() * B
                node_count += B
                
                ed_time = datetime.now()
                total_time += (ed_time - st_time).total_seconds()
            
            print("Finished training epoch:", epoch, "total time:", total_time)

            if node_count == 0:
                total = pos = neg = 0.0
            else:
                total = total_wsum / node_count
                pos = pos_wsum / node_count
                neg = neg_wsum / node_count

            print(epoch, total, pos, neg)

        self.cpu()


# evaluation function
def get_score(model, data):
    dt = data.t()
    dt1 = torch.tensor([model.mapping[x.item()] for x in dt[0]], device=device)
    dt2 = torch.tensor([model.mapping[x.item()] for x in dt[1]], device=device)
    vector1 = model.locations[dt1]
    vector2 = model.locations[dt2]
    dis = model.distance(vector1, vector2).float()
    pp = model.p_test(torch.mean(dis,dim=-1), dt1, dt2)
    return pp


# utils 
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True

def parser_add_argument(parser): 
    parser.add_argument("--h", type=int, default=17)
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--neg", type=int, default=1)
    parser.add_argument("--alpha", type=int, default=24)
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)  # 现在作为“每 batch 最大边数”
    parser.add_argument("--dataset", type=str, default='cora')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split_ratio", type=float, default=0.1)
    parser.add_argument("--c", type=int, default=1)
    parser.add_argument("--turn", type=int, default=0)
    parser.add_argument("--lam", type=float, default=1)
    parser.add_argument("--save_score", type=str, default=None)
    parser.add_argument("--save_emb", type=str, default=None)


if __name__ == "__main__":
    # 1. create the parser
    parser = argparse.ArgumentParser()
    
    # 2. add hyperparameters
    parser_add_argument(parser)

    # 3. parse the arguments
    args = parser.parse_args()
    dataset_name = args.dataset
    h = args.h
    tp = args.tp
    neg = args.neg
    gamma = args.gamma
    alpha = args.alpha
    epoch = args.epoch
    batch_size = args.batch_size
    seed = args.seed
    split_ratio = args.split_ratio
    turn = args.turn
    lam = args.lam
    c = args.c
    save_score = args.save_score
    save_emb = args.save_emb
        
    # 4. set seed
    set_random_seed(seed)
    
    # 5. load data
    print(f">>> split_ratio: {split_ratio} >>>")   
    train_data = get_train(dataset_name, str(split_ratio))
    test_data = get_test(dataset_name, str(split_ratio))
    
    # 6. train model
    model = CritiGraph(
        h=h, tp=tp, c=c, eps=1e-5, neg=neg, 
        gamma=gamma, lam=lam, alpha=alpha, epoch=epoch, batch_size=batch_size,
        save_score=save_score
    ).cuda()
    
    print("current_time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f">>> begin training >>>")
    
    model(train_data.cuda())
    
    print("current_time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f">>> finished training, start testing >>>")

    # 7. get score
    pos, neg = get_score(model, test_data[0]), get_score(model, test_data[1])
    
    if save_score is not None:
        np.save(f"{save_score}/pos.npy", pos.cpu().numpy())
        np.save(f"{save_score}/neg.npy", neg.cpu().numpy())
    
    if save_emb is not None:
        mapping = model.mapping
        num_old_nodes = max(mapping.keys()) + 1
        out = torch.full((num_old_nodes,), -1, dtype=torch.long)
        for old, new in mapping.items():
            out[old] = new
        torch.save(
            {
                'degree': model.degree.cpu(),
                'table': model.distance_lookup_table.cpu(),
                'emb': model.locations.cpu(), 
                'mapping': out,
                'gamma': model.gamma,
                'alpha': model.alpha
            },
            f"{save_emb}/emb.pt"
        )

    print("current_time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f">>> finished getting embedding, start testing >>>")