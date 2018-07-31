# pylint: disable=R,C,E1101
import torch
from functools import lru_cache
from s2cnn.utils.decorator import show_running


def so3_integrate(x):
    """
    Integrate a signal on SO(3) using the Haar measure
    
    :param x: [..., beta, alpha, gamma] (..., 2b, 2b, 2b)
    :return y: [...] (...)
    """
    assert x.size(-1) == x.size(-2)
    assert x.size(-2) == x.size(-3)

    b = x.size(-1) // 2

    w = _setup_so3_integrate(b, device_type=x.device.type, device_index=x.device.index)  # [beta]

    x = torch.sum(x, dim=-1).squeeze(-1)  # [..., beta, alpha]
    x = torch.sum(x, dim=-1).squeeze(-1)  # [..., beta]

    sz = x.size()
    x = x.view(-1, 2 * b)
    w = w.view(2 * b, 1)
    x = torch.mm(x, w).squeeze(-1)
    x = x.view(*sz[:-1])
    return x


@lru_cache(maxsize=32)
@show_running
def _setup_so3_integrate(b, device_type, device_index):
    import lie_learn.spaces.S3 as S3

    return torch.tensor(S3.quadrature_weights(b), dtype=torch.float32, device=torch.device(device_type, device_index))  # (2b) [beta]  # pylint: disable=E1102
