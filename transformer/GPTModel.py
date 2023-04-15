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
        self.Activate   = nn.GELU()
        self.OutLayer   = nn.Linear(4 * inEmbedDim, inEmbedDim, bias=inEnableBias)

    def forward(self, inX):
        # (EmbedDim, 4 * EmbedDim) -> (4 * EmbedDim, EmbedDim)
        #                   FCLayer->OutLayer
        inX = self.Activate(self.FCLayer(inX))
        inX = self.OutLayer(inX)
        return inX

class GPTBlock(nn.Module):
    def __init__(self, inEmbedDim, inHeadNum, inPosEmbedDim, inEnableBias, inEpsNorm):
        super().__init__()
        #LayerNorm 防止梯度爆炸与消失,从而加速训练过程并提高模型性能
        self.AttenLayerNorm = nn.LayerNorm(inEmbedDim, inEpsNorm)
        #自注意力机制允许模型根据输入序列中各个单词之间的关系来计算每个单词的表示
        self.AttenLayer     = CausalSelfAttention(inEmbedDim, inHeadNum, inPosEmbedDim)
        self.MLPLayerNorm   = nn.LayerNorm(inEmbedDim, inEpsNorm)
        self.MLPLayer       = GPTBlockMLP(inEmbedDim, inEnableBias)

    def forward(self, inX):
        # 自注意力机制生成新单词,与输入(inX)相加形成了ResNet
        ResNetAtt = inX + self.AttenLayer(self.AttenLayerNorm(inX))
        # MLP 层的目的是对从自注意力层传递过来的信息进行更深入的非线性处理。
        # 这有助于提取更高级别的特征，从而提高模型的表示能力
        ResNetMLP = inX + self.MLPLayer(self.MLPLayerNorm(ResNetAtt))

        return ResNetMLP
    
def GPTBlockList(inEmbedDim, inHeadNum, inPosEmbedDim, inEnableBias, inEpsNorm, inBlockNum):
    return nn.ModuleList(
        [GPTBlock(inEmbedDim, inHeadNum, inPosEmbedDim, inEnableBias, inEpsNorm) for _ in range(inBlockNum)]
    )

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.VocabSize is not None
        assert config.PosEmbedDim is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # Token Embeding (nBatchNum, t, nEmbedDim)
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
        print("\nInit Model All Weights:")
        self.apply(self.InitModelWeights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        # 对Causal Self Attention的权重进行特殊初始化
        print("\nInit AttOutLayer.weight:")
        for name, param in self.named_parameters():
            if name.endswith('AttOutLayer.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * config.BlockNum))

    # 权重的初始值对模型训练的收敛速度和最终性能有很大影响
    # 初始化权重
    def InitModelWeights(self, inModule):
        if isinstance(inModule, nn.Linear):
            torch.nn.init.normal_(inModule.weight, mean=0.0, std=0.02)
            if inModule.bias is not None:
                torch.nn.init.zeros_(inModule.bias)
        elif isinstance(inModule, nn.Embedding):
            torch.nn.init.normal_(inModule.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        print(idx)
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

    def TraversePrintModuleInfo(self) :
        for name, module in self.named_modules():
            print(f"{name}: {type(module)}")

    def TraversePrintParameters(self) :
        for name, param in self.named_parameters():
            print(f"{name}: {type(param)}: {param.nelement()}")

    def PrintNumParameters(self) :
        # report number of parameters
        print("number of parameters: %d" % (sum(p.nelement() for p in self.parameters())))
