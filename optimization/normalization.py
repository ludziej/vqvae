import torch.nn as nn


norm_resolver = {
    "none": lambda dim, _: (lambda x: x),
    "layer": lambda dim, _: nn.LayerNorm(dim, elementwise_affine=True),
    "group": lambda dim, ng: nn.GroupNorm(num_groups=ng, num_channels=dim, affine=True),
    "batch": lambda dim, _: nn.BatchNorm1d(dim, affine=True),
}


class CustomNormalization(nn.Module):
    def __init__(self, dim, norm_type="none", num_groups=32):
        super().__init__()
        self.dim = dim
        self.norm_type = norm_type
        self.norm = norm_resolver[norm_type](dim, num_groups)

    def forward(self, x):
        return self.norm(x)
