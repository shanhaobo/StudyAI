# 盲区：`.detach()` / `torch.no_grad()` / `requires_grad=False` 不是同一件事

**主题大类**：反向传播 / autograd

## 误区

三者都是"不算梯度"，互相替换无所谓。GAN 代码里 `FakeData.detach()` 写对了但说不清为什么不能换成 `no_grad()` 或 `requires_grad=False`。

## 真相

三者作用在**不同位置**：

| 写法 | 切的是什么 | 典型用途 |
|---|---|---|
| `tensor.detach()` | 把单个 tensor **从计算图剪下来**，但其值仍参与下游计算 | GAN 训 D 时把 G 的输出剪掉，让梯度不回传 G |
| `with torch.no_grad():` | 该作用域内**所有新建 tensor 都不进图**（不记录 op） | Eval / 计算指标 / EMA 更新 |
| `param.requires_grad = False` | 该参数**永久退出训练**，optimizer 不更新它 | 冻结预训练 backbone、迁移学习 |

混用的真实后果：

- 训 GAN D 时若用 `with torch.no_grad(): FakeData = G(noise)`，**G 的前向连图都没建**，G 的损失也不能再回传——D 的训练倒是 OK，但接下来训 G 时拿到的 `FakeData` 是 detached 的常量，G 的 loss `.backward()` 报错或梯度全 0。
- `requires_grad=False` 用在中间 activation 上无效（activation 的 requires_grad 由输入决定，不是用户设的）。
- `detach()` 不影响后续操作的 grad——只要 detach 后的 tensor 又参与了一个 `requires_grad=True` 的 op，新图照样建。

## 直观比喻

> `detach()` = **剪断单根线**（这一根之后的图不连原图）。
> `no_grad()` = **关掉摄像机**（这段时间发生的事一概不录）。
> `requires_grad=False` = **把这个零件焊死**（永远不参与训练）。

## 一句话记住

> **训练中切局部梯度用 `detach()`；评估/EMA 用 `no_grad()`；冻结参数用 `requires_grad=False`。** 三个动词管的是不同的"层"，别互换。
