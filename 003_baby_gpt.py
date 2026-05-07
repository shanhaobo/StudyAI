import os

import torch
from torch.nn import functional as F

from KongMing.Modules.Transformer.GPTModel import GPTModel, GPTConfig
from KongMing.Utils.OutputPath import BuildOutputPath


###################################################################################################

# 教学脚本：演示一个 4 层 4 头 GPT 在"训练前 vs 训练后"的下一 token 概率转移图。
# 用一个固定周期序列 "0110" 当训练语料；context=3 时正好 4 个上下文有数据：
#   "011"→0  "110"→0  "100"→1  "001"→1
# 其余 4 个上下文（"000/010/101/111"）训练数据里没出现，模型在那里只能靠先验输出。
# 训前画一次图：所有上下文输出几乎均匀（~50/50）。
# 训后画一次图：4 个见过的上下文锁定到 ~100%，未见过的不变。

ContextLength    = 3
TrainSequenceStr = "0110" * 64                  # 256 个 token 的周期序列
TrainSteps       = 500
LearningRate     = 0.01

OutputDir = BuildOutputPath(__file__)


###################################################################################################

def AllPossible(inN : int, inK : int):
    """枚举长度 K、字母表 [0, N) 的所有序列。"""
    if inK == 0:
        yield []
        return
    for I in range(inN):
        for Suffix in AllPossible(inN, inK - 1):
            yield [I] + Suffix


def BuildTrainingPairs(inSequenceStr : str, inContextLength : int):
    """滑窗构造 (X, y)：X 是 context_length 个 token，y 是下一个 token。"""
    Tokens = torch.tensor([int(C) for C in inSequenceStr], dtype=torch.long)
    Length = Tokens.numel() - inContextLength
    X = torch.stack([Tokens[I : I + inContextLength] for I in range(Length)])
    Y = torch.stack([Tokens[I + inContextLength]      for I in range(Length)])
    return X, Y


def PlotTransitionGraph(inGPT : GPTModel, inOutPath : str, inTitle : str):
    """对所有 context_length-bit 输入跑一次 forward，画 graphviz 转移图。
    每条边的 label 是"该 next token 的概率%"。
    """
    from graphviz import Digraph

    Dot = Digraph(comment=inTitle, engine="circo")
    Dot.attr(label=inTitle, labelloc="t", fontsize="20")

    inGPT.eval()
    with torch.no_grad():
        for Xi in AllPossible(inGPT.config.VocabSize, inGPT.config.PosEmbedDim):
            X      = torch.tensor(Xi, dtype=torch.long).unsqueeze(0)
            Logits = inGPT(X)
            Probs  = F.softmax(Logits, dim=-1)[0].tolist()
            print("input {} ---> {}".format(Xi, ["{:.3f}".format(P) for P in Probs]))

            CurrSig = "".join(str(D) for D in Xi)
            Dot.node(CurrSig)
            for T in range(inGPT.config.VocabSize):
                NextSig = "".join(str(D) for D in (Xi[1:] + [T]))
                Label = "{}({:.0f}%)".format(T, Probs[T] * 100)
                Dot.edge(CurrSig, NextSig, label=Label)

    # graphviz 渲染需要系统级 dot 可执行文件；没装时优雅降级——只存 .gv 源，
    # 用户后续装上 graphviz 再 `dot -Tpng baby_gpt_xxx.gv -o ...` 即可。
    # render() 失败前会先把 .gv 源写到 inOutPath（无扩展名），失败时把它改名成 .gv 即可。
    from graphviz.backend.execute import ExecutableNotFound
    try:
        Dot.render(inOutPath, format="png", cleanup=True)
        print("[plot] wrote {}.png".format(inOutPath))
    except ExecutableNotFound:
        SrcPath = inOutPath
        GvPath  = "{}.gv".format(inOutPath)
        if os.path.exists(SrcPath):
            os.replace(SrcPath, GvPath)
        else:
            Dot.save(GvPath)
        print("[plot] graphviz 'dot' executable not found; only saved {} (install Graphviz to render PNG)".format(GvPath))


def TrainBabyGPT(inGPT : GPTModel, inX : torch.Tensor, inY : torch.Tensor,
                 inSteps : int, inLearningRate : float) -> None:
    """简单 full-batch 训练——数据集只有 ~256 个样本，整批喂没压力。"""
    Optimizer = torch.optim.AdamW(inGPT.parameters(), lr=inLearningRate)
    inGPT.train()
    for Step in range(inSteps):
        Logits = inGPT(inX)
        Loss   = F.cross_entropy(Logits, inY)
        Optimizer.zero_grad()
        Loss.backward()
        Optimizer.step()
        if (Step + 1) % 50 == 0 or Step == 0:
            print("step {:>4d} | loss {:.4f}".format(Step + 1, Loss.item()))


###################################################################################################

if __name__ == "__main__":
    torch.manual_seed(1337)

    Config = GPTConfig(
        PosEmbedDim = ContextLength,
        VocabSize   = 2,
        BlockNum    = 4,
        HeadNum     = 4,
        EmbedDim    = 16,
        EnableBias  = False,
    )
    GPT = GPTModel(Config)
    GPT.PrintNumParameters()

    print("==== 训练前转移图（应接近均匀分布） ====")
    PlotTransitionGraph(GPT, os.path.join(OutputDir, "baby_gpt_untrained"),
                        "Baby GPT (untrained)")

    print("\n==== 在周期序列 '0110...' 上训练 ====")
    print("训练数据 4 个上下文应锁定为：011→0  110→0  100→1  001→1")
    X, Y = BuildTrainingPairs(TrainSequenceStr, ContextLength)
    TrainBabyGPT(GPT, X, Y, TrainSteps, LearningRate)

    print("\n==== 训练后转移图（见过的 4 个上下文应接近 100%） ====")
    PlotTransitionGraph(GPT, os.path.join(OutputDir, "baby_gpt_trained"),
                        "Baby GPT (trained on 0110...)")
