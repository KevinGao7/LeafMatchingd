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



class NODE2KET_Scorer(ScoringAlgorithm):
    def __init__(self, edges2scores_path):
        self.name = "node2ket"
        self.edges2scores: Dict[Edge, float] = np.load(edges2scores_path, allow_pickle=True).item()


    def log1p_mapping(self, x: np.ndarray, label: str):
        """
        将输入 ndarray 映射到 [0, 1]，使用 Log1p 变换。
        要求：若 min(x) < -1e-3 则报错（说明含有真实负值）
        """
        x = np.asarray(x)

        mn = np.min(x)
        if mn < -1e-3:
            raise ValueError(f"Detected real negative values: min={mn} at {label} scores.")

        # 修正微小浮点误差
        if mn < 0:
            x = x - mn      # 将 min 拉到 0（误差 < 1e-3 时安全）

        # Log1p 变换
        y = np.log1p(x)

        # 映射到 [0, 1]
        y = y / np.max(y) if np.max(y) > 0 else y

        return np.power(y, 0.5)


    def score_pairs(self, edge_index: np.ndarray, remap: np.ndarray) -> np.ndarray:
        assert edge_index.ndim == 2 and edge_index.shape[1] == 2

        scores = []
        for u, v in edge_index:
            e = (u, v)
            s = self.edges2scores.get(e, 0.0)   # 默认未出现的边得分为 0
            scores.append(s)
        scores = np.asarray(scores, dtype=np.float32)

        # Log1p 映射到 [0, 1]
        scores = self.log1p_mapping(scores, label="node2ket")

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
    figsize: Tuple[float,float] = (12,3)
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
def plot_panel_1x4(matrices, col_titles, cfg: VizConfig, M_obs=None):
    """
    matrices: list of 4 np.ndarray (n,n) [GT, LF, N2V, Poincare]
    col_titles: list of 4 strings
    M_obs: np.ndarray (n,n) in {0,1}, 0 = observed edge, 1 = not observed
           if None -> no overlay
    """
    def draw_obs_boundaries(ax, M_obs, color="red", linewidth=1.5):
        """
        在 heatmap 上绘制 M_obs == 0 的像素块外边界（4-邻接连通），完全对齐 imshow 网格。
        
        参数：
            ax : matplotlib Axes
            M_obs : (n,n) np.ndarray, 0 表示已观察到的边，1 表示未观察
            color : 边框颜色
            linewidth : 线宽（可自行调整）
        """
        M = np.asarray(M_obs)
        assert M.ndim == 2 and M.shape[0] == M.shape[1]
        n = M.shape[0]

        # 遍历每一个 cell
        for i in range(n):
            for j in range(n):
                if M[i, j] != 0:
                    continue  # 只给 M==0 的区域画边框

                # (i, j) 对应 imshow 中的像素中心
                # 上下左右四条边框线的坐标（注意 origin="upper" 的坐标系）
                # x 轴是列 j，对应水平；y 轴是行 i，对应垂直
                
                x0, x1 = j - 0.5, j + 0.5
                y0, y1 = i - 0.5, i + 0.5

                # 上边
                if i == 0 or M[i-1, j] != 0:
                    ax.plot([x0, x1], [y0, y0], color=color, linewidth=linewidth)

                # 下边
                if i == n-1 or M[i+1, j] != 0:
                    ax.plot([x0, x1], [y1, y1], color=color, linewidth=linewidth)

                # 左边
                if j == 0 or M[i, j-1] != 0:
                    ax.plot([x0, x0], [y0, y1], color=color, linewidth=linewidth)

                # 右边
                if j == n-1 or M[i, j+1] != 0:
                    ax.plot([x1, x1], [y0, y1], color=color, linewidth=linewidth)

    def plot_obs_points(ax: matplotlib.axes.Axes, M_obs, color='red', size=60, marker='o'):
        """
        在已有 heatmap 上，于 M_obs==1 的位置绘制点（居中）

        参数：
        ax     : matplotlib.axes.Axes
        M_obs  : 2D ndarray，0-1矩阵
        color  : 点颜色
        size   : 点大小
        marker : 点形状
        """
        ys, xs = np.where((M_obs == 0))
        ax.scatter(xs, ys, s=size, c=color, marker=marker, edgecolors=color, linewidths=0.65)


    gamma = 1.0
    matrices = [apply_gamma(M, gamma) for M in matrices]

    assert len(matrices) == 4
    n = matrices[0].shape[0]

    fig, axes = plt.subplots(1, 4, figsize=cfg.figsize)
    im = None

    for i, ax in enumerate(axes):
        M = matrices[i]
        im = ax.imshow(
            M, cmap=cfg.cmap, vmin=cfg.vmin, vmax=cfg.vmax,
            origin="upper", interpolation="nearest"
        )
        ax.set_title(col_titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        # ---- overlay: draw red borders for observed edges ----
        if M_obs is not None:
            M_obs[np.eye(M_obs.shape[0], dtype=bool)] = 1.0  # 自环不考虑观测状态
            # h, w = M_obs.shape
            # X, Y = np.meshgrid(np.arange(0, w, 0.5), np.arange(0, h, 0.5))
            # draw_obs_boundaries(ax, M_obs, color="#ba2b32", linewidth=1)
            # overlay_obs_mask(ax, M_obs, alpha=0.2, color=(1, 0, 0))
            # ax.countour(M_obs, levels=[0.5], colors="#ba2b32", linewidths=1)
            plot_obs_points(ax, M_obs, color="#a5220b", size=1.16, marker='s')
            # plot_obs_markers(ax, M_obs, color="#ba2b32", size=0.3)




    if cfg.title:
        fig.suptitle(cfg.title, fontsize=12)

    # ---- shared colorbar ----
    if cfg.show_colorbar and im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        ticks = np.linspace(0, 1, 6)
        positions = ticks ** gamma
        cbar.set_ticks(positions)
        cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
        cbar.set_label("distance (original scale)", rotation=90)

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
    remap: Optional[np.ndarray] = None,
    M_obs: Optional[np.ndarray] = None
) -> dict:
    """
    只可视化单个 ratio:
    假设 nodes 已经是【全局聚类 + 截断】后的最终节点顺序，
    所以此处不再进行任何层次化聚类。
    """

    if len(nodes) < 2:
        raise RuntimeError("Too few nodes selected for visualization")
    if cfg.verbose:
        print(f"[run] dataset={dataset_name}, ratio={ratio}, nodes={len(nodes)}")

    # ---- 1) GT SPD 按既定节点顺序计算 ----
    # gtM, h = spd_distance_matrix(G, nodes, divisor=cfg.spd_divisor, diag_value=cfg.diag_value)

    # ---- 2) 模型距离矩阵，保持相同顺序 ----
    modelMs = []
    for name, model in models:
        M = model_distance_matrix_single(model, nodes, diag_value=cfg.diag_value, remap=remap)
        modelMs.append((name, M))

    # ---- 3) 不再聚类，不再重排 ----
    # nodes 的顺序已经是最终顺序 → 直接使用
    order = np.arange(len(nodes))   # identity permutation

    # gtM_r = gtM  # no reordering

    # 模型矩阵保持顺序一致
    ordered_modelMs = modelMs  # no reordering

    # ---- 4) 构建 4-panel 对应矩阵 ----
    name_to_M = {name.lower(): M for name, M in ordered_modelMs}
    panel_mats = [
        # gtM_r,
        name_to_M.get("lf",      np.ones(len(nodes))),
        name_to_M.get("nerd",    np.ones(len(nodes))),
        name_to_M.get("headnet", np.ones(len(nodes))),
        name_to_M.get("node2ket",np.ones(len(nodes))),
    ]
    col_titles = ["LF", "Nerd", "HeadNet", "Node2KET"]

    # ---- 5) 输出文件名与 title 处理 ----
    if not cfg.out_path:
        cfg = replace(cfg, out_path=f"{dataset_name}_r{ratio}.pdf")
    if not cfg.title:
        cfg = replace(cfg, title=f"{dataset_name}, ratio={ratio}")

    # ---- 6) 绘制 ----
    plot_panel_1x4(panel_mats, col_titles, cfg, M_obs=M_obs)

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
datasets =['cora', 'male', 'cora_ml']
ratios = [0.3, 0.6, 0.9]
max_nodes = 48
seed = 1
cluster_method = 'ward'  # 'average' or 'ward'
dis_method = 'spd'    # 'spd' or 'adjacency'
# 原始 colormap
ori_cmap = plt.get_cmap("PuBu_r")
# ori_cmap = plt.get_cmap("YlOrBr")
import matplotlib.colors as mcolors
new_cmap = mcolors.LinearSegmentedColormap.from_list(
    'PuBu_r_shallow', ori_cmap(np.linspace(0.3, 1.0, 256))
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
            threshold=8
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

    print("Building observation mask...")
    M_obs = build_observation_mask(nodes, np.concatenate([train_edges_max, test_edges_max], axis=0))

    plot_observation_heatmap(
        M_obs,
        out_path=f"./visualization/{dataset}_observability.png",
        title=f"{dataset} observability (ratio={ratios[-1]:.1f})",
        cmap=cmap      # or "Blues_r"
    )

    print("Starting visualization per ratio...")
    for ratio, G, remap in zip(ratios, graphs_per_ratio, remap_per_ratio):
        cfg = VizConfig(
            max_nodes=max_nodes, 
            out_path=f"./visualization/{dataset}_r{ratio}.svg", 
            title=f"{dataset}, ratio={ratio}", 
            verbose=True,
            cluster_method=cluster_method,
            cmap=cmap
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
        )),
        ("node2ket", NODE2KET_Scorer(
            edges2scores_path=f"/home/gaochi/LF_baseline/any2ket_pybind/logs/{dataset}/r{ratio}-s{seed}/edges2scores_dict.npy"
        )),
        ]
        result = run_visualization_single_ratio(dataset, ratio, G, models, cfg, nodes=nodes, remap=remap, M_obs=M_obs)
        print("done")



