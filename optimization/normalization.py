import torch.nn as nn
import torch


class CustomNormalization(nn.Module):
    def __init__(self, dim, use_groupnorm=False, num_groups=32):
        super().__init__()
        self.dim = dim
        self.use_groupnorm = use_groupnorm
        self.norm = nn.LayerNorm(dim) if not self.use_groupnorm else \
            nn.GroupNorm(num_channels=dim, num_groups=num_groups)

    def forward(self, x):
        return self.norm(x)