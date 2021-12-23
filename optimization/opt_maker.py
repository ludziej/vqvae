
import torch as t
import old_ml_utils.dist_adapter as dist
from torch.nn.parallel import DistributedDataParallel
from old_ml_utils.fp16 import FusedAdam, FP16FusedAdam, LossScalar


def get_lr_scheduler(opt, lr_use_linear_decay, lr_scale, lr_warmup, lr_start_linear_decay, lr_decay, lr_gamma, **params):
    def lr_lambda(step):
        if lr_use_linear_decay:
            curr_lr_scale = lr_scale * min(1.0, step / lr_warmup)
            decay = max(0.0, 1.0 - max(0.0, step - lr_start_linear_decay) / lr_decay)
            if decay == 0.0:
                if dist.get_rank() == 0:
                    print("Reached end of training")
            return curr_lr_scale * decay
        else:
            return lr_scale * (lr_gamma ** (step // lr_decay)) * min(1.0, step / lr_warmup)

    shd = t.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    return shd


def get_optimizer(model, beta1, beta2, lr, weight_decay, eps, fp16_loss_scale, fp16_scale_window, fp16, fp16_opt, **params):
    # Optimizer
    betas = (beta1, beta2)
    if fp16_opt:
        opt = FP16FusedAdam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    else:
        opt = FusedAdam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

    # lr scheduler
    shd = get_lr_scheduler(opt, **params)

    # fp16 dynamic loss scaler
    scalar = None
    if fp16:
        rank = dist.get_rank()
        local_rank = rank % 8
        scalar = LossScalar(fp16_loss_scale, scale_factor=2 ** (1./fp16_scale_window))
        if local_rank == 0:
            print(scalar.__dict__)

    return opt, shd, scalar
