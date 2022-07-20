import torch


def rms_normalize(sig, rms_level=-18, eps=1e-7):
    if torch.mean(sig) < eps:
        return sig
    r = 10 ** (rms_level / 10.0)
    a = torch.sqrt(1e-7 + (len(sig) * r ** 2) / torch.sum(sig ** 2))
    y = sig * a
    return y
