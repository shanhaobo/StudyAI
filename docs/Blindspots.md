# 知识盲区索引

记录学习 StudyAI 过程中暴露出的概念性误区。每条都对应 `docs/Blindspots/<分类>/<topic>.md` 里的详细说明。

每条格式：**主题** — 误以为 X，实际 Y。

---

## 🤖 AI / 训练相关

### 训练 / 优化

- [优化器与调度器是有状态的](Blindspots/AI/优化器与调度器是有状态的.md) — 误以为优化器是无状态函数，实际 Adam/SGD-w-momentum 都带 `m/v/velocity` 等惯性记忆；LRScheduler 也带 `last_epoch`。**checkpoint 必须三件套：权重 + 优化器 + scheduler**。
- [梯度累积 ≠ 大 batch](Blindspots/AI/梯度累积不等于大Batch.md) — 误以为 8 步累积 batch=4 等价于 batch=32，实际对优化器等价、对 BatchNorm/Dropout 不等价。LayerNorm/GroupNorm 不受影响。

### 反向传播 / autograd

- [`detach()` / `no_grad()` / `requires_grad=False` 不同](Blindspots/AI/detach与no_grad与requires_grad.md) — 三者切的是不同位置：单 tensor 的图、整段作用域的图、参数的训练资格。GAN 训 D 时只能用 `detach()`。

### 数值稳定性 / 精度

- [混合精度训练为什么需要 GradScaler](Blindspots/AI/混合精度与梯度缩放.md) — 误以为 fp16 训练就是 `.half()`，实际 fp16 指数范围窄、梯度会下溢/上溢，必须 `autocast + GradScaler`；bf16 不需要 scaler。

### 模型结构 / 表达能力

- [BatchNorm 是双人格层](Blindspots/AI/BatchNorm的运行时状态.md) — 误以为 BN 训练/评估行为一致，实际两种模式用的统计量不同；`requires_grad=False` 不能冻结 BN 的 running stats，必须 `.eval()`。

### 训练工程实践

- [可复现训练需要锁全部 RNG](Blindspots/AI/可复现性需要RNG.md) — 误以为 `torch.manual_seed` 就够，实际 Python/NumPy/CUDA/DataLoader/cuDNN 是 5 个独立随机源；完全可复现还需要 checkpoint 存 RNG state。

---

## 🛠 非 AI（语言 / OS / 通用工程）

### Python / OS

- [Windows 下 Python 多进程是 spawn 不是 fork](Blindspots/非AI/Python-Windows多进程spawn模式.md) — 误以为 `multiprocessing / DataLoader(num_workers>0)` 跨平台一致，实际 Windows 用 spawn 重新 import 整个脚本，必须配 `if __name__ == "__main__"` 守卫。

---

## 维护约定

- 当对话中暴露出新的概念性盲区，落到 `Blindspots/<AI|非AI>/<主题>.md`，并在本索引追加一行。
- 分类原则：
  - **AI/**：和神经网络、训练机制、autograd、模型层语义、训练数值稳定性、训练复现性等**只在 AI 场景才会遇到**的偏差。
  - **非AI/**：Python 语言坑、OS 行为差异、通用算法、文件系统、版本控制等**学其它东西也会用到**的偏差。即便首次撞上是 AI 场景，也算非 AI（更通用）。
- 模糊地带（如"可复现性"，跨多个层）：放在主要触发领域。复现性 100% 出现在训练，所以放 AI；多进程出现在所有 Python 后端，所以放非 AI。
- "误以为 X，实际 Y" 这个对偶必须写出来——只写"什么是 Adam"等于普通教材，没有锚点。
- 不重复落已经在 `CLAUDE.md` 或 `docs/KongMing-Review.md` 记过的"工程坑"——盲区档案只放**思维模型层面的偏差**。

## 标识来源

- **实际暴露**：用户对话中明确显示出的偏差（如 [优化器有状态]）。
- **预填**：常见同类型偏差，作者尚未撞过——其余条目都是这一类，等真撞上时把"触发场景"补上日期。
