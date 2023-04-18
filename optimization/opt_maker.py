import torch as t
import logging
from torch.optim import AdamW
from optimization.ema import EMAOptimizer


def get_lr_scheduler(opt, lr_use_linear_decay, lr_scale, lr_warmup, lr_start_linear_decay, lr_decay, lr_gamma,
                     lr_decay_thresh=0, **params):
    def lr_lambda(step):
        if lr_use_linear_decay:
            curr_lr_scale = lr_scale * min(1.0, step / lr_warmup)
            decay = max(0.0, 1.0 - max(0.0, step - lr_start_linear_decay) / lr_decay)
            if decay == 0.0:
                logging.info("Reached end of training")
            return curr_lr_scale * decay
        else:
            return lr_scale * max(lr_decay_thresh, lr_gamma ** (step // lr_decay)) * min(1.0, step / lr_warmup)

    shd = t.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return shd


def get_optimizer(model, beta1, beta2, lr, weight_decay, eps, with_ema=False, ema_decay=0.9999, **params):
    # Optimizer
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2), eps=eps)

    if with_ema:
        opt = EMAOptimizer(opt, decay=ema_decay, device=model.device)

    # lr scheduler
    sch_config = {
        'scheduler': get_lr_scheduler(opt, **params),
        'interval': 'step',
        'frequency': 1,
    }
    return opt, sch_config
