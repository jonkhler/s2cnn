#pylint: disable=R,C,E1101
import torch
from functools import lru_cache

def so3_integrate(x):
    """
    :param x: [..., beta, alpha, gamma] (..., 2b, 2b, 2b)
    :return y: [...] (...)
    """
    assert x.size(-1) == x.size(-2)
    assert x.size(-2) == x.size(-3)

    b = x.size(-1) // 2

    w = setup_so3_integrate(b, x) # [beta]

    x = torch.sum(x, dim=-1).squeeze(-1) # [..., beta, alpha]
    x = torch.sum(x, dim=-1).squeeze(-1) # [..., beta]

    sz = x.size()
    x = x.view(-1, 2 * b)
    w = w.view(2 * b, 1)
    x = torch.mm(x, w).squeeze(-1)
    x = x.view(*sz[:-1])
    return x

@lru_cache(maxsize=32)
def setup_so3_integrate(b, like):
    import lie_learn.spaces.S3 as S3

    return like.new_tensor(S3.quadrature_weights(b)) # (2b) [beta]
