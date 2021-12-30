import torch.distributed as dist
from enum import Enum


class ReduceOp(Enum):
    SUM = 0,
    PRODUCT = 1,
    MIN = 2,
    MAX = 3

    def ToDistOp(self):
        return {
            self.SUM: dist.ReduceOp.SUM,
            self.PRODUCT: dist.ReduceOp.PRODUCT,
            self.MIN: dist.ReduceOp.MIN,
            self.MAX: dist.ReduceOp.MAX
        }[self]


def is_available():
    return dist.is_available()


def get_rank():
    if is_available():
        return _get_rank()
    else:
        return 0


def all_reduce(tensor, op=ReduceOp.SUM):
    if is_available():
        return _all_reduce(tensor, op)
    #else: do nothing


def _get_rank():
    return dist.get_rank()


def _all_reduce(tensor, op):
    return dist.all_reduce(tensor, op.ToDistOp())
