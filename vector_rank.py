import numpy as np


def score_vectors(code_vecs, nl_vecs, max_rank=100):
    scores = np.matmul(nl_vecs, code_vecs.T)

    ranks = np.argsort(scores, axis=-1, kind="quicksort", order=None)[:, :-max_rank-1:-1]

    return scores, ranks


def correct_rank(ranks, labels=None):
    if labels is None:
        labels = np.arange(ranks.shape[0])

    ps = np.argwhere(ranks == labels[:, np.newaxis])

    correct_rank = np.full(ranks.shape[0], np.nan)

    correct_rank[ps[:, 0]] = ps[:, 1]

    return correct_rank


def calc_mrr(correct_ranks, trunc_ranks=True):
    rr = 1 / (correct_ranks + 1)
    # if trunc_ranks:
    #     rr[r >= 100] = 0

    rr[np.isnan(rr)] = 0

    return np.mean(rr)