# 盲区：BatchNorm 是有"训练态/评估态"双人格的层

**主题大类**：模型结构 / 表达能力

## 误区

BN 就是个"标准化层"，训练和评估行为应该一致。`model(x)` 在训练和 eval 时输出不一样肯定是 bug。

## 真相

BN 维护**两份不同的统计量**，训练态和评估态用的不是同一份：

| 模式 | 用什么标准化 | 是否更新 running stats |
|---|---|---|
| `model.train()` | **当前 batch** 的 mean/var | ✅ 通过 momentum 更新到 buffer |
| `model.eval()` | buffer 里累积的 **running_mean / running_var** | ❌ 只读 |

由此衍生四个常见坑：

1. **忘了 `model.eval()` 就推理** → 用单样本 batch 的 mean/var 标准化自己，输出永远是 0 附近的常数。
2. **`batch_size=1` 训练** → 当前 batch 的 var=0，标准化后除零。要么换 GroupNorm/LayerNorm，要么 `track_running_stats=False`。
3. **冻结 backbone 微调忘了 `.eval()`** → 即使 `requires_grad=False`，BN 的 running stats 仍在被新数据污染——backbone 的"知识"被悄悄改写。`.eval()` 才能真正冻结。
4. **多 GPU 同步**：每个 GPU 的 BN 看到的是本卡的子 batch，统计量不一致。需要 `nn.SyncBatchNorm` 或换 GroupNorm。

`running_mean / running_var / num_batches_tracked` 都是 buffer，会进 state_dict——**checkpoint 隐式包含了"训练分布的痕迹"**。换数据集微调时这部分若不重置，行为会带前任的影子。

## 直观比喻

> BN 是个**双重人格**：训练时凭"当下感觉"工作（用 batch 统计），并把每天的感觉写进日记（buffer）；评估时不再相信当下，只翻日记。**忘切模式 = 在错误人格下工作**。

## 一句话记住

> **`.train()` / `.eval()` 不是装饰品，是 BN/Dropout 的开关**。`requires_grad=False` 不能替代 `.eval()`——前者管参数更新，后者管前向行为。
