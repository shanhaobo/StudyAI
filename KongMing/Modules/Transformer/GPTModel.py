import torch
import torch.nn as nn
import math

from .Transformer import Transformer

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

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.VocabSize is not None
        assert config.PosEmbedDim is not None
        self.config = config

        self.transformer = Transformer(
            config.VocabSize,
            config.EmbedDim,
            config.PosEmbedDim,
            config.HeadNum,
            config.BlockNum,
            config.EpsNorm,
            config.EnableBias
        )
        self.lm_head = nn.Linear(config.EmbedDim, config.VocabSize, bias=False)
        self.transformer._WordTknEmbed.weight = self.lm_head.weight

        # init all weights
        self.apply(self.__InitModelWeights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        # 对Causal Self Attention的权重进行特殊初始化
        for name, param in self.named_parameters():
            if name.endswith('AttOutLayer.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * config.BlockNum))

    # 权重的初始值对模型训练的收敛速度和最终性能有很大影响
    # 初始化权重
    def __InitModelWeights(self, inModule):
        if isinstance(inModule, nn.Linear):
            torch.nn.init.normal_(inModule.weight, mean=0.0, std=0.02)
            if inModule.bias is not None:
                torch.nn.init.zeros_(inModule.bias)
        elif isinstance(inModule, nn.Embedding):
            torch.nn.init.normal_(inModule.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.BlockNum, f"Cannot forward sequence of length {t}, block size is only {self.config.BlockNum}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # transformer进行计算
        x = self.transformer(idx, pos)
        # 只关心最后一个时间步[:,-1,:],因为这是模型预测的下一个词
        #[-1,:,:]最后一个矩阵:  代表最后一个样本的所有时间步
        #[:,-1,:]最后一个向量:  代表每个样本的最后一个时间步
        #[:,:,-1]最后一个标量:  代表每个样本,所有时间步,的最后一个值
        logits = self.lm_head(x[:, -1, :]) # note: only returning logits at the last time step (-1), output is 2D (b, vocab_size)
        return logits

    def SumParameters(self) :
        return sum(p.nelement() for p in self.parameters())

    #遍历Model,打印信息
    def TraversePrintModuleInfo(self) :
        for name, module in self.named_modules():
            print(f"{name}: {type(module)}")

    #遍历参数,打印信息
    def TraversePrintParameters(self) :
        for name, param in self.named_parameters():
            print(f"{name}: {type(param)}: {param.nelement()}")

    #汇总参数个数
    def PrintNumParameters(self) :
        # report number of parameters
        print("number of parameters: %d" % self.SumParameters())
