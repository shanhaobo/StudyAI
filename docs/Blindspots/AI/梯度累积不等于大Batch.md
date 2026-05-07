# 盲区：梯度累积 ≠ 大 batch（BN/Dropout 骗不过去）

**主题大类**：训练 / 优化

## 误区

显存不够想用大 batch 时，把"梯度累积 N 步"当成 "等价于 batch=N×原 batch"。

## 真相

**对优化器是等价的，对网络层不是。**

- 优化器看到的是 N 步累积的梯度均值——这部分确实等价。
- 但 BatchNorm 的 `running_mean / running_var` 是**每步前向**根据**当前 mini-batch** 计算的。8 个 batch=4 累积 vs 1 个 batch=32：
  - 前者 BN 看到 8 份"小样本统计"，方差被低估、均值噪声大。
  - 后者 BN 看到 1 份"大样本统计"，更接近真实分布。
- Dropout、SyncBN、layer-wise gradient clipping 同理——任何**依赖 batch 内统计或随机性**的层都会偏离。

LayerNorm / GroupNorm / RMSNorm 不依赖 batch 统计，所以不受影响——这就是 ViT/LLM 时代大家爱用 LayerNorm 的隐藏理由之一。

## 直观比喻

> 梯度累积是"分 8 次拍照后再洗印"，大 batch 是"一次拍 8 倍曝光的照片"。优化器只看洗出来的照片，BN 看的是相机的**测光过程**——分 8 次的话每次都基于小光圈在测。

## 一句话记住

> **梯度累积 ≈ 大 batch 仅对无 batch 内统计的层成立。** KongMing 现在用的 GroupNorm/LayerNorm 居多，所以基本安全；一旦换 BatchNorm 又用累积，就要警觉。
