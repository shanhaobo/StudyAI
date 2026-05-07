# 盲区：可复现训练 ≠ 设个 `torch.manual_seed`

**主题大类**：工程实践

## 误区

脚本顶上加一句 `torch.manual_seed(42)`，跑两次结果对不上 → 怀疑硬件随机或 PyTorch 有 bug。

## 真相

PyTorch 训练里有**至少 5 个独立的随机源**，要全部锁住才能复现：

| 随机源 | 锁的方法 |
|---|---|
| Python 内置 `random` | `random.seed(42)` |
| NumPy | `np.random.seed(42)` |
| PyTorch CPU | `torch.manual_seed(42)` |
| PyTorch CUDA（每张卡独立） | `torch.cuda.manual_seed_all(42)` |
| DataLoader worker | 每个 worker 通过 `worker_init_fn` 单独 seed |

**还不够**——cuDNN 的某些算子默认走"自动选最快"路径，每次选的可能不同：

```python
torch.backends.cudnn.deterministic = True   # 强制选确定性算子
torch.backends.cudnn.benchmark = False      # 关掉自动 benchmark
torch.use_deterministic_algorithms(True)    # PyTorch 1.8+ 全局开关
```

代价：**慢 10-30%**，部分 op（如 `index_add` GPU 版）没有确定性实现会直接报错。

`inc` 重训为什么常常完全不一样：

- 即使种子设好，**RNG state 是随训练演化的**（每次采样消耗它）。
- `torch.save` 默认不保存 `torch.get_rng_state() / torch.cuda.get_rng_state_all()`。
- 重启时 RNG 回到 seed 初态 → DataLoader 的 shuffle、Dropout mask、随机数据增强**全部走另一条路径**——即便权重和优化器状态都恢复了，下一步训练轨迹也不可能完全一样。

完整复现需要存的"第四件套"：**权重 + 优化器 + scheduler + RNG state**。

## 直观比喻

> 种子 = **从同一站台出发**，但车开起来后走哪条路（RNG state 演化）、是否换乘（cuDNN 选 op）都还是变量。要让两次行程完全一样，得连**驾驶日志**都备份。

## 一句话记住

> **完全可复现 = seed 全平台 + cuDNN 确定模式 + checkpoint 含 RNG state**。任何一环漏了都只是"看起来差不多"。这一条 KongMing 目前没做，参见 `docs/KongMing-Review.md` 第三档。
