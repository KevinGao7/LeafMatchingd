import torch
from copy import deepcopy
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import numpy as np


def eval_hits(prob_pos: torch.Tensor, prob_neg: torch.Tensor, K=100):
    if len(prob_neg) < K:
        return 1.0
    kth_score_in_negative_edges = prob_neg[-K]
    hitsK_right = torch.searchsorted(prob_pos, kth_score_in_negative_edges, right=True)
    histK_left = torch.searchsorted(prob_pos, kth_score_in_negative_edges, right=False)
    hitsK = (hitsK_right + histK_left)/2
    hitsK_score = float(1 - hitsK/len(prob_pos))
    return hitsK_score

def eval_mrr(prob_pos: torch.Tensor, prob_neg: torch.Tensor):
    inds_opt = torch.searchsorted(prob_neg, prob_pos, right=True)
    mr_opt = 1/(len(prob_neg) - inds_opt + 1)
    mrr_opt = torch.mean(mr_opt)
    inds_pess = torch.searchsorted(prob_neg, prob_pos, right=False)
    mr_pess = 1/(len(prob_neg) - inds_pess + 1)
    mrr_pess = torch.mean(mr_pess)

    return float(mrr_opt), float(mrr_pess)

def eval_auc(prob_pos: torch.Tensor, prob_neg: torch.Tensor):
    prob_pos_numpy = prob_pos.detach().cpu().numpy()
    prob_neg_numpy = prob_neg.detach().cpu().numpy()

    prob_all = np.concatenate([prob_pos_numpy, prob_neg_numpy])
    true_all = np.concatenate([np.ones(len(prob_pos_numpy)), np.zeros(len(prob_neg_numpy))]).astype(np.int32)

    rocauc = roc_auc_score(true_all, prob_all)
    ap = average_precision_score(true_all, prob_all) # also PRAUC

    sorted_prob = deepcopy(prob_all)
    sorted_prob.sort()
    threshold = sorted_prob[-len(prob_pos_numpy)]
    pred_all = (prob_all >= threshold).astype(np.int32)

    f1 = f1_score(true_all, pred_all)

    return rocauc, ap, f1

from typing import Tuple, List, Dict

class NodeMRREvaluator:
    r"""
    Node-level MRR evaluator — simplified dynamic-negative version
    ---------------------------------------------------------------
    Parameters
    ----------
    num_nodes  : int      — number of nodes in graph
    block_size : int      — number of positive edges per evaluation block
    neg_ratio  : int      — number of negatives per positive edge
    """
    def __init__(self):
        self.run = False

    def activate(self):
        self.run = True
        print("Activated NodeMRREvaluator.")

    def init(self, num_nodes: int, block_size: int = 16384, neg_ratio: int = 100):
        self.N = num_nodes
        self.block_size = block_size
        self.neg_ratio = neg_ratio

    @torch.no_grad()
    def evaluate(
        self,
        test_pos: torch.Tensor,        # [E,2] or [2,E]
        test_sco: torch.Tensor,        # [E]
        score_fn                      # callable(u,v)->score
    ) -> float:
        if test_pos.shape[0] == 2:
            test_pos = test_pos.t() # [E,2]
        E = test_pos.size(0)
        device = test_pos.device

        total_rr_opt, total_rr_pess = 0.0, 0.0
        count_opt, count_pess = 0, 0

        for i in range(0, E, self.block_size):
            pos_block = test_pos[i:i+self.block_size]
            pos_scores = test_sco[i:i+self.block_size]
            B = pos_block.size(0)
            u = pos_block[:, 0] # [B]
            v_pos = pos_block[:, 1]

            # --- negative samples ---
            v_neg = torch.randint(low=0, high=self.N, size=(B * self.neg_ratio, ), device=device)
            neg_scores = score_fn(
                torch.stack(
                    [u.unsqueeze(1).repeat(1, self.neg_ratio).reshape(-1), 
                    v_neg]
                , dim=1) # [B * neg_ratio, 2]
            ).reshape(B, self.neg_ratio)  # [B * neg_ratio] -> [B, neg_ratio]
            neg_sorted = torch.sort(neg_scores, dim=1).values  # ascending

            # --- searchsorted for rank opt ---
            inds_opt = torch.searchsorted(neg_sorted, pos_scores.unsqueeze(1), right=True)
            rank = (self.neg_ratio - inds_opt + 1).to(torch.float32)
            rr = 1.0 / rank
            total_rr_opt += rr.sum().item()
            count_opt += B

            # --- searchsorted for rank pess ---
            inds_pess = torch.searchsorted(neg_sorted, pos_scores.unsqueeze(1), right=False)
            rank = (self.neg_ratio - inds_pess + 1).to(torch.float32)
            rr = 1.0 / rank
            total_rr_pess += rr.sum().item()
            count_pess += B

        mrr_opt = total_rr_opt / count_opt
        mrr_pess = total_rr_pess / count_pess
        return mrr_opt, mrr_pess
    
MRR_NODE = NodeMRREvaluator()