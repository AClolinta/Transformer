
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()

    def forward(self, src, tgt):
        out = self.transformer(src, tgt)
        out = self.out(out)
        return out