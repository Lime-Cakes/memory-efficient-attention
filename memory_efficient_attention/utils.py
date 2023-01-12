import torch


from torch import Tensor
from typing import List

def dynamic_slice(
    x: Tensor,
    starts: List[int],
    sizes: List[int],
) -> Tensor:
    slicing = [slice(start, start + size) for start, size in zip(starts, sizes)]
    return x[slicing]


def map_pt(f, xs):
    t = [f(x) for x in xs]
    return tuple(map(torch.stack, zip(*t)))


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, torch.stack(ys)
