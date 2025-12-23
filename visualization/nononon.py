# -*- coding: utf-8 -*-
"""
无向图链接预测评测（按“阈值曲线下的面积”定义 + Precision–Recall 曲线）
- 正样本：test_edges
- 负样本：由 train 节点诱导完全图，去除 train_edges 与 test_edges（无向、无自环）
- 每个算法需提供：name 属性 与 score_pairs(edge_index) 方法，返回 [0,1] 分数
- 评测内容：
    1) 四条阈值曲线：TP/total_P, FP/total_P, FN/total_N, TN/total_N，并计算曲线面积
    2) Precision–Recall 曲线 + Average Precision (PR-AUC)
"""

from dataclasses import dataclass
from enum import unique
from typing import Iterable, List, Tuple, Dict, Sequence, Optional
import numpy as np
import torch
from tqdm.notebook import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score
from collections import defaultdict as ddict 
from itertools import count

Edge = Tuple[int, int]  # 无向图以 (min(u,v), max(u,v)) 表示

import gensim
import torch
import sys
from headnet_evaluation_utils import read_edgelist_our, load_embedding_for_evaluation, get_scores

class ScoringAlgorithm:
    """算法需实现：
        - 属性 name: str
        - 方法 score_pairs(edge_index) -> np.ndarray[(E,), float] in [0,1]
    """
    name: str
    def score_pairs(self, edge_index: np.ndarray) -> np.ndarray:
        raise NotImplementedError

# -----------------------------
# 可选：基于节点 embedding 的余弦相似度 -> 概率映射打分器
# -----------------------------
class LF_Scorer(ScoringAlgorithm):
    """
    示例：给定节点 embedding，使用余弦相似度经温度缩放映射到 [0,1]。
    score(u,v) = sigmoid( (cos(u,v) - bias) / tau )
      - tau: 温度，越小越陡
      - bias: 偏置，用于校准分数分布
    """
    def __init__(self, path: str, **kwargs):
        self.name = "ours"
        data = torch.load(path)  
        self.locations = data['locations']
        self.bucket_tensor = data['bucket']
        self.distance_lookup_table = data['table']
        self.in_degree = data['in_degree']
        self.out_degree = data['out_degree']
        self.gamma = data['gamma']
        self.alpha = data['alpha']
        self.eps = data['eps']

    def p_test(self, dis, ig1, ig2):
        deg1, deg2 = self.out_degree[ig1], self.in_degree[ig2]   
        ap = (deg1+1)*(deg2+1)
        aa = (dis+self.eps)/torch.log(ap)
        return 1/(1+aa**self.gamma/self.alpha)

    def score_pairs(self, edge_index: np.ndarray, remap: np.ndarray) -> np.ndarray:
        assert edge_index.ndim == 2 and edge_index.shape[1] == 2
        edge_index = remap[edge_index]
        
        edge_index = torch.tensor(edge_index)
        node1 = self.bucket_tensor[edge_index[:, 0]]
        node2 = self.bucket_tensor[edge_index[:, 1]]
        
        v1 = self.locations[node1]
        v2 = self.locations[node2]
        dis = self.distance_lookup_table[torch.bitwise_xor(v1, v2)].float()
        scores = self.p_test(torch.mean(dis, dim=-1), node1, node2)
        return scores.numpy()

class NERD_Scorer(ScoringAlgorithm):
    def __init__(self, hub_path: str, aut_path: str, **kwargs):
        self.name = "nerd"
        hub = self.read_w2v_emb(hub_path, binary=False)
        aut = self.read_w2v_emb(aut_path, binary=False)

        # gensim KeyedVectors
        node_keys = hub.index_to_key
        node_ids = sorted(map(int, node_keys))   # 转为 int，用于 lookup

        self.max_id = max(node_ids)

        # 构建 lookup: int_node_id -> row index
        self.lookup = np.full(self.max_id + 1, -1, dtype=np.int32)
        for row, nid in enumerate(node_ids):
            self.lookup[nid] = row

        # embedding 矩阵对齐
        dim = hub.get_vector(node_keys[0]).shape[0]
        self.hub = np.zeros((len(node_ids), dim), dtype=np.float32)
        self.aut = np.zeros_like(self.hub)

        for row, nid in enumerate(node_ids):
            k = str(nid)
            self.hub[row] = hub[k]
            self.aut[row] = aut[k]

    def sigmoid(self, x):
        out = np.empty_like(x)

        pos = x >= 0
        neg = ~pos

        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[neg])
        out[neg] = exp_x / (1.0 + exp_x)

        return out

    
    def read_w2v_emb(self, file_path, binary):
        return gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=0)
    
    def score_pairs(self, edge_index: np.ndarray, remap: np.ndarray) -> np.ndarray:
        assert edge_index.ndim == 2 and edge_index.shape[1] == 2
        edge_index = remap[edge_index]
        
        scores = self.sigmoid(
            np.sum(
                self.hub[self.lookup[edge_index[:, 0]]] *
                self.aut[self.lookup[edge_index[:, 1]]],
                axis=-1
            )
        )
        
        return scores


class HEADNET_Scorer(ScoringAlgorithm):
    def __init__(self, path: str):
        self.name = "headnet"
        self.embedding = load_embedding_for_evaluation('klh', path)


    def negval_to_score_exp(
        self, x, tau=None, clip_quantile=0.99, eps=1e-12
    ):
        """
        将非正数 x 映射到 (0,1]，使用 s = exp(x / tau)

        x: np.ndarray 形状 (N,)
            通常为负相似度、负损失等
        tau: float or None
            衰减尺度，若为 None，则取 -median(x)
        clip_quantile: float
            裁剪尾部，避免极端值导致 exp 下溢
        """
        x = torch.as_tensor(x, dtype=torch.float64)

        # 负值右侧尾部裁剪（例如最小的 1%）
        if clip_quantile is not None:
            q = torch.quantile(x, 1 - clip_quantile)   # 注意这里是右尾
            x = torch.maximum(x, q)

        if tau is None:
            tau = -torch.median(x)     # 使 x/tau ≈ O(1)
        tau = max(float(tau), eps)

        s = torch.exp(x / tau)
        return s.numpy()

    def score_pairs(self, edge_index: np.ndarray, remap: np.ndarray) -> np.ndarray:
        assert edge_index.ndim == 2 and edge_index.shape[1] == 2
        edge_index = remap[edge_index]
        
        ori_scores = get_scores(self.embedding, edge_index, 'klh')
        scores = self.negval_to_score_exp(ori_scores, tau=None, clip_quantile=0.99, eps=1e-12)
        
        return scores
    
def get_edges(path: str) -> np.ndarray:
    with open(path, 'r') as f:
        edges = [tuple(map(int, line.strip().split(sep='\t'))) for line in f]
    return edges



import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Sequence, Optional, Callable, Any
from dataclasses import dataclass

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# -----------------------------
# Configuration defaults
# -----------------------------

# 单 cell：对单个 ratio 输出一张 1x4 面板（GT | LF | Node2vec | Poincaré）
# 依赖：numpy, networkx, scipy, matplotlib
# 前提：LF_Scorer, Node2vec_Scorer, Poincare_Scorer, get_edges 已在 notebook 中定义

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Any
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# ---------- 配置类 ----------
@dataclass
class VizConfig:
    topk: int = 30               # 选取节点数量（按 degree）
    spd_divisor: float = None    # SPD 归一化除数；None -> 用选中节点的最大有限距离
    cluster_method: str = "ward"  # SciPy linkage method 用 average（GT 距离）
    diag_value: float = 0.0
    cmap: str = "Blues_r"
    vmin: float = 0.0
    vmax: float = 1.0
    show_colorbar: bool = True
    out_path: str = "panel.png"
    title: str = ""   # e.g. "cora, ratio=0.1"
    figsize: Tuple[float,float] = (12,3)
    verbose: bool = False

# ---------- 小工具 ----------
def build_graph_from_edges(edges: Sequence[Tuple[int,int]]) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G

def adjacency_distance_matrix(G: nx.DiGraph, nodes):
    n = len(nodes)
    M = np.ones((n, n), dtype=np.float32)
    idx = {u: i for i, u in enumerate(nodes)}
    for u in nodes:
        for v in G.successors(u):
            if v in idx:
                M[idx[u], idx[v]] = 0.0
    np.fill_diagonal(M, 0.0)
    return M   # non-symmetric if directed


def select_common_nodes_across_ratios(graphs, topk):
    """
    graphs: list of nx.Graph, one per ratio (sorted by ratio ascending)
    topk: desired number of nodes
    reference graph = graphs[-1]  (largest ratio)
    """
    # 1. 取所有图节点交集
    common = set(graphs[0].nodes())
    for G in graphs[1:]:
        common &= set(G.nodes())

    # 2. 交集数量检查
    if len(common) < topk:
        raise RuntimeError(
            f"[FATAL] Node intersection size = {len(common)}, smaller than topk={topk}. "
            "Cannot guarantee cross-ratio comparability."
        )

    # 3. 按最大 ratio 图的度排序
    ref = graphs[-1]
    deg = dict(ref.degree(common))
    sorted_nodes = sorted(common, key=lambda x: (-deg[x], x))
    return sorted_nodes[:topk]


def all_pairs_ordered(nodes: Sequence[int]) -> np.ndarray:
    # 返回 shape (n*n,2) 的 node pairs（有序行优先）
    nodes = np.asarray(nodes, dtype=np.int64)
    n = nodes.shape[0]
    a = np.repeat(nodes, n)
    b = np.tile(nodes, n)
    return np.stack([a, b], axis=1)

def reshape_to_square(vals: np.ndarray, n: int) -> np.ndarray:
    return vals.reshape((n,n))

# ---------- 距离矩阵计算 ----------
def spd_distance_matrix(G: nx.Graph, nodes: Sequence[int], divisor=None, diag_value=0.0) -> np.ndarray:
    # 计算 shortest path lengths，仅基于 nodes。不可达设为 1.0。归一化到 [0,1]
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    n = len(nodes)
    M = np.ones((n,n), dtype=np.float32)
    finite_ds = []
    for i,u in enumerate(nodes):
        li = lengths.get(u, {})
        for j,v in enumerate(nodes):
            if u==v:
                continue
            d = li.get(v, None)
            if d is not None:
                finite_ds.append(float(d))
    if divisor is None:
        maxd = max(finite_ds) if finite_ds else 1.0
        norm_div = 10 #maxd if maxd>0 else 1.0
    else:
        norm_div = divisor if divisor>0 else 1.0
    for i,u in enumerate(nodes):
        li = lengths.get(u, {})
        for j,v in enumerate(nodes):
            if i==j:
                M[i,j] = diag_value
                continue
            d = li.get(v, None)
            if d is None:
                M[i,j] = 1.0
            else:
                val = float(d)/norm_div
                M[i,j] = min(val, 1.0)
    return M

def model_distance_matrix_single(model: Any, nodes: Sequence[int], diag_value=0.0, remap=None) -> np.ndarray:
    # model 必须实现 score_pairs(edge_index: np.ndarray)->np.ndarray in [0,1]
    pairs = all_pairs_ordered(nodes)  # shape (n*n,2)
    scores = model.score_pairs(pairs, remap)  # expected shape (n*n,)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1 or scores.size != pairs.shape[0]:
        raise RuntimeError("model.score_pairs must return flat array of length n*n")
    dists = 1.0 - scores
    n = len(nodes)
    M = reshape_to_square(dists, n)
    np.fill_diagonal(M, diag_value)
    np.clip(M, 0.0, 1.0, out=M)
    return M

# ---------- 聚类排序（使用 GT） ----------
def hierarchical_order(M: np.ndarray, method="average") -> np.ndarray:
    """
    M: 真实距离矩阵，可以是非对称的（有向最短路或模型距离）
    这里仅用于计算层次聚类的顺序，所以先用 max 对称化。
    """
    # 用 max 做对称化，仅用于聚类，不改原矩阵
    A = np.minimum(M, M.T).copy()
    # A = (M + M.T) / 2.
    
    np.fill_diagonal(A, 0.0)

    y = squareform(A, checks=False)
    Z = linkage(y, method=method)
    return leaves_list(Z)


def apply_gamma(M: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(np.clip(M, 0, 1), gamma)

# ---------- 绘图（1x4 panel） ----------
def plot_panel_1x4(matrices, col_titles, cfg: VizConfig):
    # matrices: list of 4 np.ndarray (n,n) [GT, LF, N2V, Poincare]
    gamma=1.0
    matrices = [apply_gamma(M, gamma) for M in matrices]
    
    assert len(matrices)==4
    n = matrices[0].shape[0]
    fig, axes = plt.subplots(1,4, figsize=cfg.figsize)
    im = None
    for i, ax in enumerate(axes):
        M = matrices[i]
        im = ax.imshow(M, cmap=cfg.cmap, vmin=cfg.vmin, vmax=cfg.vmax, origin="upper", interpolation="nearest")
        ax.set_title(col_titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    if cfg.title:
        fig.suptitle(cfg.title, fontsize=12)
    # --- Colorbar with correct gamma ticks ---
    if cfg.show_colorbar and im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        ticks = np.linspace(0,1,6)
        positions = ticks ** gamma
        cbar.set_ticks(positions)
        cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
        cbar.set_label("distance (original scale)", rotation=90)
    # fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(cfg.out_path, dpi=200, bbox_inches="tight")
    if cfg.verbose:
        print(f"Saved {cfg.out_path}")
    plt.close(fig)

# ---------- 主运行函数（单 ratio） ----------
from dataclasses import replace

def run_visualization_single_ratio(
    dataset_name: str,
    ratio: float,
    G: str,
    models: List[Tuple[str, Any]],   # list of (name, model_obj), model_obj must implement score_pairs
    cfg: VizConfig,
    nodes: Optional[List[int]] = None,
    spd_ref_matrix: Optional[np.ndarray] = None,
    remap: Optional[np.ndarray] = None
) -> dict:
    """
    只可视化单个 ratio:
      - dataset_name: 用于 title 和文件名
      - ratio: number, 用于 title 和文件名
      - train_edges_path: path -> get_edges(...)
      - models: [("lf", LF_Scorer(...)), ("node2vec", Node2vec_Scorer(...)), ("poincare", Poincare_Scorer(...))]
    输出文件名 cfg.out_path（若未设置，将自动设为 "{dataset}_r{ratio}.png"）
    """

    if len(nodes) < 2:
        raise RuntimeError("Too few nodes selected for visualization")
    if cfg.verbose:
        print(f"[run] dataset={dataset_name}, ratio={ratio}, nodes={len(nodes)}")
    # Compute GT SPD
    gtM = spd_distance_matrix(G, nodes, divisor=cfg.spd_divisor, diag_value=cfg.diag_value)
    # Compute model matrices (order same)
    modelMs = []
    for name, model in models:
        M = model_distance_matrix_single(model, nodes, diag_value=cfg.diag_value, remap=remap)
        modelMs.append((name, M))
    # Determine order from GT
    order = hierarchical_order(spd_ref_matrix, method=cfg.cluster_method)
    gtM_r = gtM[order][:, order]
    # reorder models
    ordered_modelMs = [(name, M[order][:, order]) for name, M in modelMs]
    # Build panel matrices: GT | LF | N2V | Poincare  (enforce column order)
    # Find LF, node2vec, poincare in provided models (if missing, still place placeholders)
    name_to_M = {name.lower(): M for name, M in ordered_modelMs}
    # fetch in expected order
    panel_mats = [gtM_r,
                  name_to_M.get("lf", np.ones_like(gtM_r)),
                  name_to_M.get("nerd", np.ones_like(gtM_r)),
                  name_to_M.get("headnet", np.ones_like(gtM_r))]
    col_titles = ["GT-SPD", "LF", "Nerd", "HeadNet"]
    # output path
    if not cfg.out_path:
        cfg = replace(cfg, out_path=f"{dataset_name}_r{ratio}.png")
    # set title if not provided
    if not cfg.title:
        cfg = replace(cfg, title=f"{dataset_name}, ratio={ratio}")
    # plot
    plot_panel_1x4(panel_mats, col_titles, cfg)
    return {"nodes": nodes, "order": order, "matrices": {"gt": gtM_r, "models": name_to_M}}


def build_observation_mask(nodes, train_edges):
    """
    nodes: list[int], final ordered node ids
    train_edges, test_edges: list[(u,v)], raw int pairs
    return: M in {0,1}, shape (n,n)
            0 = observed edge exists (in train or test)
            1 = no observed edge
    """
    n = len(nodes)
    idx = {u: i for i, u in enumerate(nodes)}

    # undirected edge set
    E = set()
    for u, v in train_edges:
        E.add((u, v))

    M = np.ones((n, n), dtype=np.float32)
    for i, u in enumerate(nodes):
        M[i, i] = 0.0
        for j, v in enumerate(nodes):
            if i != j and (u, v) in E:
                M[i, j] = 0.0
    return M


def plot_observation_heatmap(M_obs, out_path, title="", cmap="Blues_r"):
    plt.figure(figsize=(4, 4))
    plt.imshow(M_obs, cmap=cmap, vmin=0, vmax=1, origin="upper", interpolation="nearest")
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.colorbar(label="observed (0) / missing (1)")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def get_edges(dataset, ratio):
    data = torch.load(f"/home/gaochi/leaf_matching/datasets/{dataset}/split/split_dict_{ratio}.pt")
    remap = data["remap"].numpy()              # Tensor: old_id -> new_id 映射（或相反）
    uniq_nodes = data["uniq_nodes"].numpy()  # Tensor: 新编号对应的原始节点 ID 列表

    # 取出边 (已被重编号为 [0, N-1])
    train_edges = data["train"]["edge"].numpy()
    test_pos_edges = data["test"]["edge"].numpy()
    test_neg_edges = data["test"]["edge_neg"].numpy()

    # 反向映射到原始节点 ID
    train_edges = uniq_nodes[train_edges]
    test_pos_edges = uniq_nodes[test_pos_edges]
    test_neg_edges = uniq_nodes[test_neg_edges]

    return train_edges, test_pos_edges, test_neg_edges, remap, uniq_nodes



# ---------- 使用示例（供粘贴运行） ----------
# 假设你已有 LF_Scorer, Node2vec_Scorer, Poincare_Scorer, get_edges
datasets = ['cora', 'cora_ml', 'citeseer']
ratios = [0.3, 0.5, 0.7, 0.9]
topk = 128
seed = 1

for dataset in datasets:
    # 1. 先加载所有 ratio 的图
    graphs_per_ratio = []
    remap_per_ratio = []
    for ratio in ratios:
        print(f"Preparing graph for ratio={ratio}...")
        train_edges, _, _, remap, _ = get_edges(dataset, ratio)
        graphs_per_ratio.append(build_graph_from_edges(train_edges))
        remap_per_ratio.append(remap)


    # 2. 计算统一节点（== 你要的跨 ratio 对齐）
    nodes = select_common_nodes_across_ratios(graphs_per_ratio, topk=topk)
    
    # 3. 基于 ratio 最大的图计算 SPD，用作全局排序基准
    
    spd_ref_matrix = spd_distance_matrix(
        graphs_per_ratio[-1],   # 最后一个 = ratio 最大
        nodes,
        divisor=None,
        diag_value=0.0
    )


    for ratio, G, remap in zip(ratios, graphs_per_ratio, remap_per_ratio):
        cfg = VizConfig(
            topk=topk, 
            out_path=f"./visualization/{dataset}_r{ratio}.png", 
            title=f"{dataset}, ratio={ratio}", 
            verbose=True,
            cluster_method='ward'
        )
        models = [
        ("lf", LF_Scorer(
            path=f'/home/gaochi/leaf_matching/logs_oi/{dataset}/scores/s{seed}-r{ratio}/emb.pt'
        )),
        ("nerd", NERD_Scorer(
            hub_path=f"/home/gaochi/LF_baseline/nerd/emb/{dataset}/hub_r{ratio}_s{seed}.txt", 
            aut_path=f"/home/gaochi/LF_baseline/nerd/emb/{dataset}/aut_r{ratio}_s{seed}.txt"
        )),
        ("headnet", HEADNET_Scorer(
            path=f'/home/gaochi/LF_baseline/HEADNET/embeddings/{dataset}/r{ratio}_s{seed}/'
        ))
        ]
        result = run_visualization_single_ratio(dataset, ratio, G, models, cfg, nodes=nodes, spd_ref_matrix=spd_ref_matrix, remap=remap)
        print("done")

    train_edges_max, _, _, remap, _ = get_edges(dataset, ratios[-1])
    # test_edges_max  = get_edges(f"./datasets/{dataset}/split_{ratios[-1]:.1f}/test_pos.txt")

    M_obs = build_observation_mask(nodes, train_edges_max)

    plot_observation_heatmap(
        M_obs,
        out_path=f"./visualization/{dataset}_observability.png",
        title=f"{dataset} observability (ratio={ratios[-1]:.1f})",
        cmap=cfg.cmap      # or "Blues_r"
    )

