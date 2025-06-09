import numpy as np


def compute_reciprocal_rank(pred_ids, gold_ids):
    """ pred_ids is sorted. """
    if not gold_ids:
        return None
    gold_ids = set(gold_ids)
    rank = None
    for r, id_ in enumerate(pred_ids):
        if id_ in gold_ids:
            rank = r + 1
            break
    return (1 / rank) if rank else 0


def compute_average_precision(pred_ids, gold_ids):
    """ pred_ids is sorted. """
    if not gold_ids:
        return None
    gold_ids = set(gold_ids)
    precisions, curr_hit = [], 0
    for i, id_ in enumerate(pred_ids):
        if id_ in gold_ids:
            curr_hit += 1
            precisions.append(curr_hit / (i + 1))
        if curr_hit == len(gold_ids):
            break  # Early stop
    ap = (sum(precisions) / len(precisions)) if precisions else 0
    return ap


def compute_ndcg(pred_ids, goldid2score, topk=None):
    """ pred_ids is sorted. """
    topk = topk or len(pred_ids)
    if not isinstance(goldid2score, dict):
        goldid2score = {gold: 1 for gold in goldid2score}

    pred_scores = np.array([(goldid2score.get(id_, 0)) for id_ in pred_ids[:topk]])  # No assumption on pred length
    gold_scores = np.array(sorted(goldid2score.values(), reverse=True)[:topk])
    pred_dcg = (pred_scores / np.log2(np.arange(2, len(pred_scores) + 2))).sum().item()
    gold_dcg = (gold_scores / np.log2(np.arange(2, len(gold_scores) + 2))).sum().item()
    return (pred_dcg / gold_dcg) if gold_dcg else None

def compute_pair_recall(pred_ids, gold_ids):
    pred_ids, gold_ids = set(pred_ids), set(gold_ids)
    return len(gold_ids & pred_ids), len(gold_ids)


def compute_pair_precision(pred_ids, gold_ids):
    pred_ids, gold_ids = set(pred_ids), set(gold_ids)
    return len(gold_ids & pred_ids), len(pred_ids)


def compute_query_recall(pred_ids, gold_ids):
    pred_ids, gold_ids = set(pred_ids), set(gold_ids)
    return (len(gold_ids & pred_ids) / len(gold_ids)) if gold_ids else None


def compute_query_precision(pred_ids, gold_ids):
    pred_ids, gold_ids = set(pred_ids), set(gold_ids)
    return (len(gold_ids & pred_ids) / len(pred_ids)) if pred_ids else None


def compute_f_score(p, r, beta=1.0):
    beta_squared = beta ** 2
    return (1 + beta_squared) * (p * r) / ((beta_squared * p) + r) if p or r else 0


def compute_query_hit(pred_ids, gold_ids):
    pred_ids, gold_ids = set(pred_ids), set(gold_ids)
    return float(len(pred_ids & gold_ids) > 0)
