import torch.nn.functional as F
from torch import nn

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


class Conv1dWeightStandardized(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, use_standardization=True,):
        self.use_standardization = use_standardization
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        if self.use_standardization:
            weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
            weight = weight - weight_mean
            std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1) + 1e-5
            weight = weight / std.expand_as(weight)
        return F.conv1d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


