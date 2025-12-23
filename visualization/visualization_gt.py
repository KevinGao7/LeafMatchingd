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

from ast import Not
import matplotlib
from dataclasses import dataclass
from enum import unique
from typing import Iterable, List, Tuple, Dict, Sequence, Optional
import numpy as np
from pyparsing import line
import torch
from tqdm.notebook import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score
from collections import defaultdict as ddict 
from itertools import count
from matplotlib.patches import Polygon

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
    topk: int = None        # 将被弃用，但保留兼容
    max_nodes: int = 128    # 新增，严格限制节点数
    spd_divisor: float = None    # SPD 归一化除数；None -> 用选中节点的最大有限距离
    cluster_method: str = "ward"  # SciPy linkage method 用 average（GT 距离）
    diag_value: float = 0.0
    cmap: str = "Blues_r"
    vmin: float = 0.0
    vmax: float = 1.0
    show_colorbar: bool = True
    out_path: str = "panel.png"
    title: str = ""   # e.g. "cora, ratio=0.1"
    verbose: bool = False

# ---------- 小工具 ----------
def build_graph_from_edges(edges: Sequence[Tuple[int,int]]) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G



def get_common_nodes(graphs):
    """
    graphs: list of nx.Graph, one per ratio (sorted by ratio ascending)
    topk: desired number of nodes
    reference graph = graphs[-1]  (largest ratio)
    """
    common = set(graphs[0].nodes())
    for G in graphs[1:]:
        common &= set(G.nodes())
    return sorted(common)



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
import os, hashlib, pickle
import numpy as np
import networkx as nx
from tqdm import tqdm

def spd_distance_matrix(G: nx.DiGraph, nodes, divisor=None, diag_value=0.0, threshold=100):
    """
    计算最短路距离矩阵（支持缓存与并行化预取）
    缓存文件路径: ./visualization/_tmp/{hash}.pkl
    """
    os.makedirs("./visualization/_tmp", exist_ok=True)

    # ---------- 1. 缓存哈希 ----------
    h = hashlib.md5()
    h.update(str(sorted(G.edges())).encode())
    cache_path = f"./visualization/_tmp/{h.hexdigest()}.pkl"

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            lengths = pickle.load(f)
    else:
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        with open(cache_path, "wb") as f:
            pickle.dump(lengths, f)

    # ---------- 2. 预提取并加速 ----------
    h.update(threshold.to_bytes(4, byteorder='little'))
    h.update(str(sorted(nodes)).encode())
    cache_path = f"./visualization/_tmp/{h.hexdigest()}.pkl"
    
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            M = pickle.load(f)
        return M, h
    
    n = len(nodes)
    node_to_idx = {u: i for i, u in enumerate(nodes)}
    M = np.ones((n, n), dtype=np.float32)

    finite_ds = []
    for u in nodes:
        li = lengths.get(u, {})
        for v, d in li.items():
            if u != v and v in node_to_idx:
                finite_ds.append(float(d))
    if divisor is None:
        maxd = max(finite_ds) if finite_ds else 1.0
        norm_div = 10  # 或 maxd
    else:
        norm_div = divisor if divisor > 0 else 1.0

    # ---------- 3. 向量化填充 ----------
    # 收集所有可达三元组 (i,j,d)
    triplets = []
    for u in nodes:
        i = node_to_idx[u]
        li = lengths.get(u, {})
        for v, d in li.items():
            if u == v or v not in node_to_idx:
                continue
            if d > threshold:
                continue
            j = node_to_idx[v]
            triplets.append((i, j, (d - 1) / norm_div))

    if triplets:
        triplets = np.array(triplets, dtype=np.float32)
        M[triplets[:, 0].astype(int), triplets[:, 1].astype(int)] = np.minimum(triplets[:, 2], 1.0)
    np.fill_diagonal(M, diag_value)
    
    with open(cache_path, "wb") as f:
        pickle.dump(M, f)
    return M, h


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
def hierarchical_order(M: np.ndarray, method="average", h=None) -> np.ndarray:
    """
    M: 真实距离矩阵，可以是非对称的（有向最短路或模型距离）
    这里仅用于计算层次聚类的顺序，所以先用 max 对称化。
    """
    h.update(b"hierarchical_order")
    h.update(method.encode())
    cache_path = f"./visualization/_tmp/{h.hexdigest()}.pkl"
    
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            order = pickle.load(f)
        return order
    
     
    
    # 用 max 做对称化，仅用于聚类，不改原矩阵
    A = np.maximum(M, M.T).copy()
    # A = (M + M.T) / 2.
    
    np.fill_diagonal(A, 0.0)

    y = squareform(A, checks=False)
    Z = linkage(y, method=method)
    order = leaves_list(Z)
    with open(cache_path, "wb") as f:
        pickle.dump(order, f)
    return order

def hierarchical_order_from_adj(G, nodes, method="ward"):
    M = adjacency_distance_matrix(G, nodes)
    S = np.maximum(M, M.T)  # symmetric
    np.fill_diagonal(S, 0.0)
    y = squareform(S, checks=False)
    Z = linkage(y, method=method)
    return leaves_list(Z)


def apply_gamma(M: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(np.clip(M, 0, 1), gamma)


# ---------- 绘图（1x4 panel） ----------
def plot_panel_1x2(matrices, col_titles, cfg: VizConfig, fig, axes: Sequence[matplotlib.axes.Axes], 
                   istop=False, ratio=0.1):
    gamma = 1.0
    matrices = [apply_gamma(M, gamma) for M in matrices]

    im = None

    for i, ax in enumerate(axes):
        M = matrices[i]
        im = ax.imshow(
            M, cmap=cfg.cmap, vmin=cfg.vmin, vmax=cfg.vmax,
            origin="upper", interpolation="nearest"
        )
        if istop:
            ax.set_title(col_titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # if istop:
        #     ax.set_title(cfg.title, fontsize=12)

    

    if cfg.show_colorbar and im is not None:
        fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)


# ---------- 主运行函数（单 ratio） ----------
from dataclasses import replace

def run_visualization_single_ratio(
    dataset_name: str,
    ratios: List[float], graphs_per_ratio: List[nx.DiGraph], remap_per_ratio: List[np.ndarray],
    cfg: VizConfig,
    nodes: Optional[List[int]] = None,
) -> dict:
    """
    可视化所有 ratios:
    假设 nodes 已经是【全局聚类 + 截断】后的最终节点顺序，
    所以此处不再进行任何层次化聚类。
    """

    if len(nodes) < 2:
        raise RuntimeError("Too few nodes selected for visualization")
    if cfg.verbose:
        print(f"[run] dataset={dataset_name}, ratios={ratios}, nodes={len(nodes)}")

    fig, axes = plt.subplots(len(ratios), 2, figsize=(6, 3*len(ratios)))
    
    for i, (ratio, G, remap) in enumerate(zip(ratios, graphs_per_ratio, remap_per_ratio)):
        # ---- 1) GT SPD 按既定节点顺序计算 ----
        gtM, h = spd_distance_matrix(G, nodes, divisor=cfg.spd_divisor, diag_value=cfg.diag_value)

        # ---- 2) 模型距离矩阵，保持相同顺序 ----
        model = LF_Scorer(
            path=f'/home/gaochi/leaf_matching/logs_oi/{dataset_name}/scores/s1-r{ratio}/emb.pt'
        )
        M = model_distance_matrix_single(model, nodes, diag_value=cfg.diag_value, remap=remap)

        # ---- 3) 不再聚类，不再重排 ----
        # nodes 的顺序已经是最终顺序 → 直接使用
        order = np.arange(len(nodes))   # identity permutation

        gtM_r = gtM  # no reordering

        # 模型矩阵保持顺序一致
        ordered_model = M  # no reordering

        # ---- 4) 构建 4-panel 对应矩阵 ----
        name_to_M = {"lf": ordered_model}
        panel_mats = [
            gtM_r,
            name_to_M.get("lf",      np.ones(len(nodes))),
        ]
        col_titles = ["", ""]

        # ---- 6) 绘制 ----
        plot_panel_1x2(panel_mats, col_titles, cfg, fig, axes[i], istop=(i==0))

    plt.savefig(cfg.out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return {
        "nodes": nodes,
        "order": order,
        "matrices": {"models": name_to_M}
    }



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
datasets =['cora', 'male']
ratios = [0.3, 0.6, 0.9]
max_nodes = 20
seed = 1
cluster_method = 'ward'  # 'average' or 'ward'
dis_method = 'spd'    # 'spd' or 'adjacency'
# 原始 colormap
ori_cmap = plt.get_cmap("PuBu_r")
# ori_cmap = plt.get_cmap("YlOrBr")
import matplotlib.colors as mcolors
new_cmap = mcolors.LinearSegmentedColormap.from_list(
    'PuBu_r_shallow', ori_cmap(np.linspace(0.1, 1.0, 256))
)
cmap = new_cmap

for dataset in datasets:
    # 1. 先加载所有 ratio 的图
    graphs_per_ratio = []
    remap_per_ratio = []
    print(ratios)
    for ratio in ratios:
        print(f"Preparing graph for ratio={ratio}...")
        train_edges, _, _, remap, _ = get_edges(dataset, ratio)
        graphs_per_ratio.append(build_graph_from_edges(train_edges))
        remap_per_ratio.append(remap)


    # 2. 计算统一节点（== 你要的跨 ratio 对齐）
    print("Computing common nodes...")
    common_nodes = get_common_nodes(graphs_per_ratio)

    if len(common_nodes) < max_nodes:
        raise RuntimeError(
            f"[FATAL] node intersection = {len(common_nodes)} < max_nodes={max_nodes}"
        )
    # 3. 基于最大 ratio 图的 GT 距离矩阵聚类，得到全局节点顺序
    print("Computing global node order...")
    if dis_method == 'spd':
        # SPD on largest ratio graph
        print("    Computing SPD reference distance matrix...")
        spd_ref_matrix, h = spd_distance_matrix(
            graphs_per_ratio[-1],
            common_nodes,
            divisor=None,
            diag_value=0.0,
            threshold=6 if dataset != 'yeast' else 100
        )
        print("    Computing hierarchical clustering...")
        # cluster to get full order
        order = hierarchical_order(spd_ref_matrix, method=cluster_method, h=h)
    elif dis_method == 'adjacency':
        raise NotImplementedError("Adjacency-based clustering is not implemented yet.")
    
    
    print(f"Total common nodes = {len(common_nodes)}; using top {max_nodes} nodes for visualization.")
    # apply leaves_list result, truncate to max_nodes
    nodes = [common_nodes[i] for i in order[:max_nodes]]
    
    ### 保存这个 nodes，以便后续 ratio 可复用 ###
    np.save(f"./visualization/{dataset}_common_nodes.npy", np.array(nodes, dtype=np.int64))

    train_edges_max, test_edges_max, _, remap, _ = get_edges(dataset, ratios[-1])




    print("Starting visualization per ratio...")
    
    cfg = VizConfig(
        max_nodes=max_nodes, 
        out_path=f"./visualization/{dataset}_gtlf.svg", 
        title=f"{dataset}", 
        verbose=True,
        cluster_method=cluster_method,
        cmap=cmap
    )

    result = run_visualization_single_ratio(
        dataset, ratios, graphs_per_ratio, remap_per_ratio,
        cfg, nodes=nodes, 
    )
    print("done")



