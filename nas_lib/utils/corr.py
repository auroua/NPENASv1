from scipy.stats import pearsonr, spearmanr, kendalltau


def get_pearsonr_coorlection(v1, v2):
    if isinstance(v1, list):
        if len(v1) == 1:
            return 0
    else:
        if len(v1.tolist()) == 1:
            return 0
    v, p = pearsonr(v1, v2)
    return v, p


def get_spearmanr_coorlection(v1, v2):
    if isinstance(v1, list):
        if len(v1) == 1:
            return 0
    else:
        if len(v1.tolist()) == 1:
            return 0
    v, p = spearmanr(v1, v2)
    return v, p


def get_kendalltau_coorlection(v1, v2):
    if isinstance(v1, list):
        if len(v1) == 1:
            return 0
    else:
        if len(v1.tolist()) == 1:
            return 0
    v, p = kendalltau(v1, v2)
    return v, p