import torch
import torch.nn as nn
import math

from .CausalSelfAttention import CausalSelfAttention

# these are default GPT-2 hyperparameters
class GPTConfig :
    def __init__(
            self,
            VocabSize = 50304,
            PosEmbedDim = 1024,
            BlockNum = 12,
            HeadNum = 12,
            EmbedDim = 768,
            EpsNorm = 1e-5,
            EnableBias = False
        ) :
        self.VocabSize      = VocabSize
        self.PosEmbedDim    = PosEmbedDim
        self.BlockNum       = BlockNum
        self.HeadNum        = HeadNum
        self.EmbedDim       = EmbedDim
        self.EpsNorm        = EpsNorm
        self.EnableBias     = EnableBias


class GPTBlockMLP(nn.Module):
    def __init__(self, inEmbedDim, inEnableBias):
        super().__init__()
        self.FCLayer    = nn.Linear(inEmbedDim, 4 * inEmbedDim, bias=inEnableBias)
        self.OutLayer   = nn.Linear(4 * inEmbedDim, inEmbedDim, bias=inEnableBias)
        self.Activate   = nn.GELU()

    def forward(self, ioX):
        ioX = self.FCLayer(ioX)
        ioX = self.Activate(ioX)
        ioX = self.OutLayer(ioX)
        return ioX

class GPTBlock(nn.Module):
    def __init__(self, inEmbedDim, inHeadNum, inPosEmbedDim, inEnableBias, inEpsNorm):
        super().__init__()
        self.AttenLayerNorm = nn.LayerNorm(inEmbedDim, inEpsNorm)
        self.AttenLayer     = CausalSelfAttention(inEmbedDim, inHeadNum, inPosEmbedDim)
        self.MLPLayerNorm   = nn.LayerNorm(inEmbedDim, inEpsNorm)
        self.MLPLayer       = GPTBlockMLP(inEmbedDim, inEnableBias)

    def forward(self, ioX):
        ioX = ioX + self.AttenLayer(self.AttenLayerNorm(ioX))
        ioX = ioX + self.MLPLayer(self.MLPLayerNorm(ioX))
        return ioX
    
def GPTBlockList(inEmbedDim, inHeadNum, inPosEmbedDim, inEnableBias, inEpsNorm, inBlockNum):
    return nn.ModuleList([GPTBlock(inEmbedDim, inHeadNum, inPosEmbedDim, inEnableBias, inEpsNorm) for _ in range(inBlockNum)])

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.VocabSize is not None
        assert config.PosEmbedDim is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.VocabSize, config.EmbedDim),
            wpe = nn.Embedding(config.PosEmbedDim, config.EmbedDim),
            h = GPTBlockList(
                    config.EmbedDim,
                    config.HeadNum,
                    config.PosEmbedDim,
                    config.EnableBias,
                    config.EpsNorm,
                    config.BlockNum
                ),
            ln_f = nn.LayerNorm(config.EmbedDim),
        ))
        self.lm_head = nn.Linear(config.EmbedDim, config.VocabSize, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.BlockNum))

        # report number of parameters
        print("number of parameters: %d" % (sum(p.nelement() for p in self.parameters()),))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.BlockNum, f"Cannot forward sequence of length {t}, block size is only {self.config.BlockNum}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x[:, -1, :]) # note: only returning logits at the last time step (-1), output is 2D (b, vocab_size)
        return logits
