import torch.nn as nn
import torch
from vqvae.encdec import Encoder, assert_shape


class Discriminator(nn.Module):
    def __init__(self, input_channels, emb_width, level, downs_t, strides_t, **block_kwargs):
        super().__init__()
        self.encoder = Encoder(input_channels, emb_width, level + 1,
                               downs_t, strides_t, **block_kwargs)
        self.fc = nn.Linear(emb_width, 2)
        self.logsoftmax = nn.LogSoftmax()
        self.nllloss = nn.NLLLoss()

    def get_proba(self, x):
        x = self.encoder(x)

        nb, ns, se = x.shape
        logits = self.fc(x.reshape(nb * ns, se))
        probs = self.logsoftmax(logits)
        return probs

    def forward(self, x, y):
        probs = self.get_proba(x)

        nb, ns, _ = probs.shape
        assert_shape(y, (nb,))
        y_true = y.view((nb, 1)).repeat(1, ns).reshape(nb * ns)

        loss = self.nllloss(probs, y_true)
        return loss

