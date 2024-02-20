import torch


def _off_diagonal(mat):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = mat.shape
    assert n == m
    return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def DecorrLoss(x):
    eps = 1e-8
    N, C = x.shape
    if N == 1:
        return 0.0
    x = x - x.mean(dim=0, keepdim=True)
    x = x / torch.sqrt(eps + x.var(dim=0, keepdim=True))
    corr_mat = torch.matmul(x.t(), x)
    loss = (_off_diagonal(corr_mat).pow(2)).mean()
    loss = loss / N
    return loss