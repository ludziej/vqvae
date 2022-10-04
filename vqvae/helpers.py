import numpy as np
import torch as t
from utils.old_ml_utils import spectral_convergence, spectral_loss, multispectral_loss


def spectral_loss_util(forward_params, x_target, x_out):
    if forward_params.use_nonrelative_specloss:
        sl = spectral_loss(x_target, x_out, forward_params) / forward_params.bandwidth['spec']
    else:
        sl = spectral_convergence(x_target, x_out, forward_params)
    sl = t.mean(sl)
    return sl


def multispectral_loss_util(forward_params, x_target, x_out):
    sl = multispectral_loss(x_target, x_out, forward_params) / forward_params.bandwidth['spec']
    sl = t.mean(sl)
    return sl


def calculate_strides(strides, downs):
    return [stride ** down for stride, down in zip(strides, downs)]


def get_hop_lengths(strides, downs):
    return np.cumprod(calculate_strides(strides, downs))


def get_sample_len_from_tokens(strides, downs, level, tokens):
    hop_lengths = get_hop_lengths(strides, downs)
    return tokens * hop_lengths[level]


def _loss_fn(loss_fn, x_target, x_pred, hps):
    if loss_fn == 'l1':
        return t.mean(t.abs(x_pred - x_target)) / hps.bandwidth['l1']
    elif loss_fn == 'l2':
        return t.mean((x_pred - x_target) ** 2) / hps.bandwidth['l2']
    elif loss_fn == 'linf':
        residual = ((x_pred - x_target) ** 2).reshape(x_target.shape[0], -1)
        values, _ = t.topk(residual, hps.linf_k, dim=1)
        return t.mean(values) / hps.bandwidth['l2']
    elif loss_fn == 'lmix':
        loss = 0.0
        if hps.lmix_l1:
            loss += hps.lmix_l1 * _loss_fn('l1', x_target, x_pred, hps)
        if hps.lmix_l2:
            loss += hps.lmix_l2 * _loss_fn('l2', x_target, x_pred, hps)
        if hps.lmix_linf:
            loss += hps.lmix_linf * _loss_fn('linf', x_target, x_pred, hps)
        return loss
    else:
        assert False, f"Unknown loss_fn {loss_fn}"