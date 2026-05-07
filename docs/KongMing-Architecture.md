# KongMing 框架架构图 + 主流框架对比（2026-05-07）

本文档落盘三块内容：

1. **框架图**：分层架构、控制流、事件链注册顺序
2. **对比表**：与 PyTorch Lightning / HF Accelerate / fastai / Keras / Catalyst 的职责对应
3. **总评**：原创亮点 + 真正的差距 + 性价比排序的改进建议

风格说明：所有"改进建议"都遵循作者的 PascalCase + `in/b` 前缀约定，**不**把命名风格列为缺点。

---

## 一、KongMing 框架图

### 1.1 分层架构（垂直）

```
┌─────────────────────────────────────────────────────────────────────┐
│  入口层  │  001_*.py … 009_*.py                                     │
│          │  组装 ModelFactory → Executor.Train/Eval                 │
└─────────────────────────────────────────────────────────────────────┘
                       │ duck-typed 7-method 协议
                       │ NewTrain/IncTrain/Eval/LoadLastest/
                       │ IsExistModels/ForceSave../ForceExit..
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  调度层  │  Executor (KongMing/Utils/Executor.py)                   │
│          │  - argv 解析（- vs --）                                  │
│          │  - newtrain / inctrain / eval 决策                       │
│          │  - keyboard 全局热键 (Ctrl+S / Ctrl+X)                   │
└─────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  组装层  │  BaseModelFactory                                        │
│          │  ├─ SingleNNModelFactory   （单网络）                    │
│          │  └─ MultiNNModelFacotry    （GAN / Diffusion 多组件）    │
│          │  职责：装配 Trainer + Archiver、把 Trainer 的            │
│          │  Begin/End{Train,Epoch,Batch} 事件挂到 Archiver 上、     │
│          │  统一 device、stdout tee 到 train.log、热键回调入口      │
└─────────────────────────────────────────────────────────────────────┘
                       │                              │
        ┌──────────────┘                              └──────────────┐
        ▼                                                            ▼
┌──────────────────────────────────┐         ┌────────────────────────────────┐
│  训练层  BaseTrainer (abc)       │         │  持久化层  BaseArchiver        │
│  ├─ SingleNNTrainer              │         │  ├─ SingleNNArchiver           │
│  └─ MultiNNTrainer               │         │  └─ MultiNNArchiver            │
│      ├─ GANTrainer / WGANTrainer │         │  - NNModuleDict {name: Module} │
│      ├─ DDPMTrainer              │         │  - NNModuleNameOnlyForTrain    │
│      ├─ VGGTrainer / VGGMNN..    │         │  - 原子 .tmp + os.replace      │
│      └─ CodecTrainer             │         │  - weights_only=True load      │
│                                  │         │  - 按 epoch 编号               │
│  - __DontOverride__Train         │         │                                │
│  - 抽象: _CreateOptimizer        │         │  使用 FileManagerWithNum 管理  │
│           _CreateLossFN          │         │  ArchivedModels/<时间戳>/      │
│           _BatchTrain            │         │   <NNName>_<epoch>.pkl         │
│  - Delegate 事件总线 (6 个)      │         │                                │
└──────────────────────────────────┘         └────────────────────────────────┘
                       │                              │
                       └──────────────┬───────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  网络层（可持久化整网）                                             │
│  KongMing/Models/ —— BaseNNModel(nn.Module)                         │
│    ├─ self.BackPropagater (优化器/调度器/损失/EMA decay/PendingState)│
│    ├─ ApplyOptimizer / ApplyLRScheduler / ApplyLossFunc             │
│    ├─ ApplyEMA → _EMARegistry[id(target)]                           │
│    ├─ with model: …  (zero_grad / backward / step / EMA update)     │
│    └─ StateDictForArchive() = {"Model":…, "BackPropagater":{Opt,LR}}│
└─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  零件层（普通 nn.Module，不持久化）                                 │
│  KongMing/Modules/                                                  │
│    Resampling{Up,Down}/  Attentions/  Transformer/  Zoo/{UNet,EMA}/ │
│    PositionEmbedding / AveragedModule / CustomEnhancedModules       │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 控制流（一次 `python 005_DDPM.py inc`）

```
argv ──► Executor.__GetArgs/__AnalyzeArgs
         │
         ▼
Executor.Train(loader) ──► Model.IncTrain(loader, start, iter)
                           │
                           ├─► Archiver.LoadLastest()
                           │     └─► torch.load(*.pkl, weights_only=True)
                           │         └─► BaseNNModel.LoadStateDictFromArchive
                           │             ├─ self.load_state_dict(Model)
                           │             └─ BackPropagater.StashPendingState
                           │
                           └─► Trainer.Train(loader, start+1, iter)
                                 │
                                 │  __DontOverride__Train
                                 ├─ _CreateOptimizer ──► ApplyOptimizer
                                 │                       └─ TryApplyPending(Optimizer)
                                 ├─ _CreateLossFN
                                 ├─ BeginTrain  ────┐──► Factory.__BMBeginTrain
                                 │                  │   (open log, read SaveInterval/
                                 │                  │    PrintInterval/LossEMADecay)
                                 │  for epoch:      │
                                 │   __DontOverride__EpochTrain
                                 │     for batch:
                                 │       BeginBatchTrain
                                 │       _BatchTrain  ─► with self.NN as M:
                                 │                          M.CalcAndAcceptLoss
                                 │                       (退出时 backward+step+EMA)
                                 │       EndBatchTrain ─► SingleNNTrainer 打印 Loss
                                 │   EndEpochTrain ────► (1) NN.UpdateLRScheduler
                                 │                       (2) Factory.__BMEndEpochTrain
                                 │                           按 SaveInterval 落盘
                                 │   软退出/到达 EndEpoch → break
                                 └─ EndTrain ─────────► Archiver.Save 最后一帧 + 关 log
```

### 1.3 事件链（Delegate）注册顺序

```
Trainer.BeginTrain      : [Factory.__BMBeginTrain]
Trainer.BeginEpochTrain : []
Trainer.BeginBatchTrain : []
Trainer.EndBatchTrain   : [Factory.__BMEndBatchTrain,
                           SingleNNTrainer.__SNNEndBatchTrain]
Trainer.EndEpochTrain   : [SingleNNTrainer.__SNNEndEpochTrain ──► UpdateLRScheduler  ← 先
                           Factory.__BMEndEpochTrain          ──► Archiver.Save     ← 后]
Trainer.EndTrain        : [Factory.__BMEndTrain]
```
> 顺序对 `inc` 安全性很关键：调度器先 step，存盘最后。失败不会丢调度器更新。

---

## 二、和主流框架对比

按"职责对应表"先点对点画清楚，再讲总评。

| 职责 | KongMing | PyTorch Lightning | HF Accelerate | fastai | Keras / TF | Catalyst |
|---|---|---|---|---|---|---|
| 训练循环托管 | `BaseTrainer.__DontOverride__Train` | `pl.Trainer.fit` | 用户写 loop, Accelerator 包 device/AMP/DDP | `Learner.fit` | `model.fit` | `Runner.train` |
| 模型抽象 | `BaseNNModel(nn.Module)` 自带优化器/EMA | `LightningModule` 自带 `configure_optimizers` | 纯 `nn.Module` | `Learner` 包 model | `keras.Model` | 纯 `nn.Module` + Runner 注入 |
| 钩子/事件 | 6 个 Delegate（手动 `.add`） | 30+ 个 `on_*` callback | hook（少）+ 用户自写 | callback 系统（很重） | callback | callback |
| 持久化 | 自研 `BaseArchiver`，按 epoch 文件 + 时间戳目录 | `ModelCheckpoint` callback + `.ckpt` 单文件 | `accelerator.save_state` | `Learner.save` 单文件 | `model.save_weights` | `CheckpointCallback` |
| 多网络（GAN/Diffusion） | 原生 `MultiNN*`，`NNModuleNameOnlyForTrain` | 自动训练循环不友好，要写 `manual optimization` | 用户自处理 | 不强 | functional 拼 | 较强（multi-stage） |
| 优化器/调度器状态保存 | ✅（BackPropagater 协议） | ✅ | ✅ | ✅ | ✅ | ✅ |
| 设备 / DDP / FP16 | ❌ 仅 cuda/cpu，无 AMP/DDP | ✅ 一行切 | ✅（核心卖点） | ✅ | ✅ | ✅ |
| 回调失败隔离 | ❌ 抛出会中断事件链 | ✅ 内置异常包裹 | n/a | ✅ | ✅ | ✅ |
| 协议化接口 | duck typing 7 方法 | ABC + Protocol | Protocol/dataclass | 弱 | ABC | 弱 |
| 实验追踪 | ❌（只有 train.log tee） | ✅（TensorBoard/W&B/MLflow 集成） | 用户接 | ✅ | ✅ | ✅ |
| 学习曲线 | ⭐ 中（看代码就懂） | ⭐⭐⭐ 高（魔法多） | ⭐ 低 | ⭐⭐ 中 | ⭐⭐ 中 | ⭐⭐ 中 |

### 2.1 KongMing 的"原创亮点"

1. **`with model:` 上下文协议** —— 比 Lightning 的 `manual_backward` 干净。`__enter__` zero_grad、`__exit__` backward+step+EMA，配合多个网络可以分别 `with G:` `with D:` 自然嵌套。这点比 Lightning 的 multi-optimizer 方案直观很多。
2. **NNModuleDict + NNModuleNameOnlyForTrain 双名单** —— 把"训练期需要、Eval 期可丢"显式建模。Lightning 没有等价物，GAN 中 D 在 inference 时占显存只能手动处理。
3. **跨时间戳目录搜索 epoch 文件**（`FindFileAcrossAllTimestampsByEpoch`）—— 比 Lightning 的"checkpoint 必须在同一目录"更宽松，多次 inc 中断重启的工作流非常贴合。
4. **PendingState 两阶段灌注** —— Lightning 也是 load 后才 configure_optimizers，但他们处理是 hook 内部隐式完成；KongMing 显式两阶段更易调试。

### 2.2 真正的差距（按"会让你疼"的程度排序）

#### 🔴 A. 无 AMP / 无 DDP / 无梯度累积
- **现象**：CIFAR10 + VGG16 + 224×224 在 8GB 卡上很难走通；DDPM 想多卡几乎不可能。
- **对比**：HF Accelerate 改 3 行就能开 `bf16` + `accumulation_steps=4` + `multi-gpu`。
- **建议**：在 `BaseNNModel.BackPropagaterClass` 内加 `_GradScaler` + `_AccumSteps`：
  - `with model:` 的 `__exit__` 改成 `if (step+1) % accum == 0: scaler.step + scaler.update; else: 不 step`；
  - `BeginBackPropagate` 改成只在 step boundary `zero_grad`；
  - device 探测加一个 `bf16-friendly` 分支（A 卡 / 30 系以上）。
  - 不用 Accelerate 那种 monkey patch，用现有的 `BackPropagater` 包一下就行——和 KongMing 风格一致。

#### 🔴 B. Delegate 失败传播
现在一个 callback 抛异常整个事件链断掉。Lightning 的做法是 try/except + 记录，KongMing 的现状是"靠注册顺序保证 scheduler 先跑"。这是脆契约。
- **建议**：在 `Delegate.__call__` 加 `bIsolateFailure: bool = False`：True 时遍历所有 callback 收集异常 → 末尾 raise 聚合。事件链上把 IO 类（Save、log close）标 isolate。

#### 🟡 C. 没有"指标 / 验证集"概念
- **现象**：Trainer 只有 `_BatchTrain`，没有 `_BatchValid`；`Eval` 走的是另一条独立分支（`Archiver.Eval`），没有"训练中按 epoch 跑 val"。
- **对比**：Lightning `validation_step` / fastai `valid_dl` 是必备品；分类任务（VGG 系列）没 val 等于看 train loss 蒙。
- **建议**：`BaseTrainer.__DontOverride__EpochTrain` 末尾加可选 `inValidLoader`，调用 `_BatchValid`（默认实现 = 空）。`SingleNNTrainer` 在 `__SNNEndEpochTrain` 末尾打印一行 val loss / acc。这一步代价小，回报大。

#### 🟡 D. 实验配置散在入口脚本顶部
ImageSize / NumClasses / lr 等大写常量混在 `00X_*.py`，再训练一次想换超参就要改源码（不能写 git 比对）。
- **对比**：Hydra/OmegaConf 一份 yaml 就解决；Lightning 也支持 `LightningCLI`。
- **建议**：不上 yaml 框架（违反 KongMing "无配置文件" 哲学），但**入口脚本顶部那批常量统一收到一个 `Config` PascalCase dataclass**，并允许 `--XxxYyy=val` 透传覆盖（Executor 已经支持，加个映射就行）。这样既保留作者风格也获得可比性。

#### 🟡 E. NNModuleDict 注册必须手写
新增一个网络 → 同时改 Trainer.\_\_init\_\_ + Archiver 注册 + ModelFactory 装配，三处不一致就静默掉权重。
- **对比**：Lightning `self.module = X` 自动登记；HF `accelerator.prepare(model)` 一行解决。
- **建议**：`BaseNNModel.__init_subclass__` 或在 ModelFactory 装配时反射 `Trainer` 上所有 `BaseNNModel` 类型属性自动登记。**收益是省 boilerplate，但代价是失去显式声明**——是否做要看作者偏好。

#### 🟢 F. 时间戳目录隔离让"实验对比"变难
每次 `new` 训练新建一个时间戳目录是优点（不污染），但缺点是没有 `experiment_id` / `tag` 维度，人眼对比要靠时间戳。
- **建议**：`--ModelTag=Run_A` 已经能透到 `inKVArgs`（前面修过 casefold 那个 bug 就是为这准备的），让 `FileManagerWithNum` 的根目录拼 `<basename>/<dataset>/<tag>/<timestamp>/`。零破坏，加一层。

#### 🟢 G. `print` tee 到 `train.log` 但没有结构化日志
重启训练 = 多份 `train.log`，并且 grep "Loss:" 提取数字得写正则。
- **建议**：除了 tee，再额外写一行 jsonl 到 `train.metrics.jsonl`（每个 batch 一行 `{"epoch":i,"batch":j,"loss":x,"avg":y}`）。两个并存，后期画曲线只读 jsonl。代价 < 10 行。

#### 🟢 H. `keyboard` 是 Windows-only 的隐性桎梏
Linux/Mac/容器里要 root 才能挂热键。已经 try/except 了不会崩，但 **Ctrl+S/Ctrl+X 在 Linux 实际上是禁用的**。
- **建议**：写一个文件 sentinel fallback——`watch <log_dir>/.save` 文件存在则触发，删掉再监听。`touch .save` 比 root 友好。

---

## 三、总评

KongMing 的定位是 **"个人学习用、可控、可读的中型框架"**，体量介于 fastai 和"裸写 PyTorch"之间。它的**网络组件协议（BackPropagater + with 上下文 + EMA registry + NNModuleDict 二级名单）是真的优雅**，比 Lightning 的隐式魔法清楚太多。

但从**支持现有项目能跑得更顺**的角度看，按性价比该补的顺序是：

> **B（Delegate 隔离）→ A（AMP/累积）→ C（Val 钩子）→ G（jsonl 指标）**

前两个是一两小时的活，后两个一下午。剩下的 D/E/F/H 都是锦上添花，不做也不影响吃饭。
