import torch

from old_ml_utils import dist_adapter as dist


def assert_shape(x, exp_shape):
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"


def average_metrics(_metrics):
    metrics = {}
    for _metric in _metrics:
        for key, val in _metric.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(val)
    return {key: sum(vals)/len(vals) for key, vals in metrics.items()}


def allreduce(x, op=dist.ReduceOp.SUM):
    x = torch.tensor(x).float().cuda()
    dist.all_reduce(x, op=op)
    return x.item()