import torch.nn as nn
import torch
from vqvae.encdec import Encoder, assert_shape


class Discriminator(nn.Module):
    def __init__(self, input_channels, emb_width, level, downs_t, strides_t, reduce_type="max", **block_kwargs):
        super().__init__()
        self.reduce_type = reduce_type
        self.encoder = Encoder(input_channels, emb_width, level + 1, downs_t, strides_t, **block_kwargs)
        self.fc = nn.Linear(emb_width, 2)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.nllloss = nn.NLLLoss(reduction="none")

    def forward(self, x):
        x = self.encoder(x)[-1]

        x = torch.max(x, dim=2).values if self.reduce_type == "max" else torch.mean(x, dim=2)
        logits = self.fc(x)
        probs = self.logsoftmax(logits)
        return probs

    def calculate_loss(self, x, y, balance=True):
        probs = self.forward(x)

        y_weight = (y / torch.sum(y) + (1 - y) / torch.sum(1 - y))/2 if balance else 1/len(y)
        loss_ew = self.nllloss(probs, y)
        loss = torch.sum(loss_ew * y_weight)

        probs = torch.exp(probs)
        cls = torch.round(probs[:, 1])
        acc = torch.sum((cls == y) * y_weight)
        return loss, probs, cls, acc, loss_ew

