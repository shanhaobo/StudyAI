# KongMing 演进路线图（2026-05-07）

按性价比排序的可执行补丁清单。每条给出**诊断 / 改法 / 影响文件 / 估时 / 风险**五段，方便挑哪条做。

> 与 `docs/KongMing-Architecture.md`（架构图 + 对比 + 总评）互补：那篇是"为什么"，本篇是"怎么做"。

## 推荐执行顺序

```
B  Delegate 失败隔离          ── 30 分钟，零风险
A  AMP / 梯度累积 / DDP       ── 2-4 小时，中风险
C  验证集钩子 (_BatchValid)   ── 1-2 小时，低风险
G  jsonl 结构化指标日志       ── 30 分钟，零风险
─────  以上是"刚需补齐"  ─────
F  ModelTag 维度              ── 1 小时，低风险
D  Config dataclass           ── 2 小时，低风险
E  NNModuleDict 自动注册      ── 1 小时，看偏好
H  文件 sentinel fallback     ── 30 分钟，零风险
```

---

## 🔴 B. Delegate 失败隔离 ★ 最先做

### 诊断
`KongMing/Utils/Delegate.py` 当前 11 行：

```python
def __call__(self, *inArgs, **inKVArgs) -> None:
    for tFunc in self.FunctionList:
        tFunc(*inArgs, **inKVArgs)
```

任一 callback 抛异常，后续 callback 全部不执行。当前事件链上 `EndEpochTrain` 注册顺序是 `scheduler.step` 先、`Save` 后——靠"注册顺序"保证 scheduler 先跑是脆约定。一旦哪天写新 callback 把 IO 类放前面，就静默丢调度器更新。

### 改法
给 `Delegate` 加可选 `bIsolateFailure`：

```python
class Delegate:
    def __init__(self, bIsolateFailure : bool = False):
        self.FunctionList = []
        self.bIsolateFailure = bIsolateFailure

    def __call__(self, *inArgs, **inKVArgs) -> None:
        if not self.bIsolateFailure:
            for tFunc in self.FunctionList:
                tFunc(*inArgs, **inKVArgs)
            return
        Errors = []
        for tFunc in self.FunctionList:
            try:
                tFunc(*inArgs, **inKVArgs)
            except Exception as e:
                Errors.append((tFunc, e))
                print("[Delegate] callback {} raised: {}".format(getattr(tFunc, "__name__", tFunc), e))
        if Errors:
            raise RuntimeError("Delegate had {} callback failures".format(len(Errors)))
```

`BaseTrainer.__init__` 把 `EndEpochTrain` / `EndTrain` 改成 `Delegate(bIsolateFailure=True)`——这两条链上跑 IO（Save / log close），最容易出错；`Begin*` / `EndBatchTrain` 保持严格模式（早 fail 早暴露）。

### 影响文件
- `KongMing/Utils/Delegate.py`（+15 行）
- `KongMing/Trainer/BaseTrainer.py`（2 行 `Delegate()` → `Delegate(bIsolateFailure=True)`）

### 估时 / 风险
30 分钟 / 零风险。默认参数 `False` 不影响旧行为。

---

## 🔴 A. AMP / 梯度累积 / DDP

### 诊断
`BaseModelFactory.__init__` 只有 cuda/cpu 二分；`BackPropagater.EndBackPropagate` 直接 `loss.backward() + optimizer.step()`。后果：
- 8GB 卡跑 VGG16+224 OOM；
- DDPM 想多卡训练完全做不到；
- 没法用 `bf16` 在 30/40 系卡省一半显存。

### 改法（分两步走）

**第一步：AMP + 梯度累积（单卡）**
在 `BaseNNModel.BackPropagaterClass` 加：

```python
self._GradScaler   = None       # 由 ApplyAMP 创建
self._AMPDtype     = None       # torch.float16 / torch.bfloat16
self._AccumSteps   = 1
self._AccumCounter = 0

def ApplyAMP(self, inDtype = torch.bfloat16):
    self._AMPDtype = inDtype
    if inDtype == torch.float16:
        self._GradScaler = torch.cuda.amp.GradScaler()

def ApplyGradAccum(self, inSteps : int):
    self._AccumSteps = max(1, int(inSteps))
```

`BeginBackPropagate` 只在 `self._AccumCounter == 0` 时 `zero_grad`；
`EndBackPropagate` 改成：

```python
self._AccumCounter += 1
bIsBoundary = (self._AccumCounter % self._AccumSteps == 0)
Loss = self._Loss / self._AccumSteps      # 关键：均摊
if self._GradScaler is not None:
    self._GradScaler.scale(Loss).backward()
    if bIsBoundary:
        self._GradScaler.step(self._Optimizer)
        self._GradScaler.update()
        self._AccumCounter = 0
else:
    Loss.backward()
    if bIsBoundary:
        self._Optimizer.step()
        self._AccumCounter = 0
```

`BaseModelFactory.__BMBeginTrain` 读 `inKVArgs["AMP"]`（`fp16` / `bf16` / 不传）和 `inKVArgs["GradAccum"]` 推到所有 `BaseNNModel`。

**第二步：DDP（可选）**
把 `BaseModelFactory.__init__` 的 device 探测拆出 `_BuildDevice()`，DDP 模式下用 `torch.nn.parallel.DistributedDataParallel` 包 `NNModuleDict[Name]`。这一步动 `Archiver.Save/Load`（要剥 `.module` 前缀），非必要不做。

### 影响文件
- `KongMing/Models/BaseNNModel.py`（+30 行 BackPropagater 改动）
- `KongMing/ModelFactory/BaseModelFactory.py`（+10 行读 KV 并传播）
- 入口脚本可选 `--AMP=bf16 --GradAccum=4` 传参，无需改源码

### 估时 / 风险
AMP+累积 2-3 小时 / 中风险（需要在 GAN 多优化器上验证）。
DDP 单独至少 4 小时 / 高风险。**先只做 AMP+累积**。

---

## 🟡 C. 验证集钩子（_BatchValid）

### 诊断
`BaseTrainer` 只有 `_BatchTrain`，没有"训练中按 epoch 跑 val"的位置。VGG 系列在 CIFAR10 上现在只能看 train loss，分不清是过拟合还是没拟合。

### 改法
`BaseTrainer.__DontOverride__EpochTrain` 末尾加：

```python
if inValidLoader is not None:
    self.__DontOverride__EpochValid(inValidLoader, inArgs, inKVArgs)
```

新增：

```python
def __DontOverride__EpochValid(self, inValidLoader, inArgs, inKVArgs):
    self.BeginEpochValid(inArgs, inKVArgs)
    for self.CurrBatchIndex, (Data, Label) in enumerate(inValidLoader):
        with torch.no_grad():
            self._BatchValid(Data, Label, inArgs, inKVArgs)
    self.EndEpochValid(inArgs, inKVArgs)

def _BatchValid(self, inBatchData, inBatchLabel, inArgs, inKVArgs):
    pass   # 子类按需重写
```

`Train()` 签名加 `inValidLoader = None`；入口脚本 `Exec.Train(loader, valid_loader, ...)`（保持向后兼容：不传就跟现在一样）。

`SingleNNTrainer.__SNNEndEpochTrain` 末尾打印一行 `Val Loss: x | Val Acc: y`（具体 metric 由子类在 `_BatchValid` 累加到 `self.ValidMetrics`）。

### 影响文件
- `KongMing/Trainer/BaseTrainer.py`（+25 行）
- `KongMing/Trainer/VGGTrainer.py` 等（按需实现 `_BatchValid`）
- `KongMing/ModelFactory/BaseModelFactory.NewTrain/IncTrain` 新增 `inValidLoader` 透传

### 估时 / 风险
1.5 小时 / 低风险。默认 None 不破坏旧入口。

---

## 🟢 G. jsonl 结构化指标日志

### 诊断
当前 `train.log` 是 tee 的人类格式，提取数字要写正则。Loss 曲线无法直接画。

### 改法
`BaseModelFactory.__OpenLogFile` 同时打开第二个文件 `train.metrics.jsonl`：

```python
self._MetricsFile = open(os.path.join(LogDir, "train.metrics.jsonl"), "a", encoding="utf-8", buffering=1)
```

新增 `__BMEndBatchTrain`（已存在但是空的）改成：

```python
def __BMEndBatchTrain(self, inArgs, inKVArgs) -> None:
    if self._MetricsFile is None or not self.Trainer.ShouldPrintBatch():
        return
    Records = {"Epoch": self.Trainer.CurrEpochIndex,
               "Batch": self.Trainer.CurrBatchIndex,
               "Time" : datetime.now().isoformat()}
    for Name, Module in self.Archiver.NNModuleDict.items():
        if hasattr(Module, "GetLossValue"):
            Loss, Avg = Module.GetLossValue()
            Records["{}.Loss".format(Name)] = Loss
            Records["{}.AvgLoss".format(Name)] = Avg
    self._MetricsFile.write(json.dumps(Records) + "\n")
```

### 影响文件
- `KongMing/ModelFactory/BaseModelFactory.py`（+15 行）

### 估时 / 风险
30 分钟 / 零风险。新增文件，不动现有 tee。

---

## 🟢 F. ModelTag 维度

### 诊断
每次 `new` 建新时间戳目录避免污染，但没有人类可读 tag。`ArchivedModels/2026-05-07_18-30-00/` vs `ArchivedModels/2026-05-08_09-15-22/`——分不出哪个是"试 lr=1e-3"哪个是"试 lr=3e-4"。

### 改法
`Executor.__GetArgs` 已经会把 `--ModelTag=lr_3e-4` 收到 `KVArgsForML`。让它穿到 `FileManagerWithNum` 的根目录构造：

```python
# FileManagerWithNum 接受 inSubDir 参数（默认 None）
# 路径变成: ArchivedModels/<tag>/<timestamp>/  或 ArchivedModels/<timestamp>/
```

`BaseModelFactory.__init__` 在初始化前从 `inKVArgs` 取 `ModelTag` 传给 Archiver。

### 影响文件
- `KongMing/Archiver/Path/FileManagerWithNum.py`（+5 行 ModelTag 子目录）
- `KongMing/Archiver/BaseArchiver.py`（构造时透传）
- `KongMing/ModelFactory/BaseModelFactory.py`（构造时读 KV）

### 估时 / 风险
1 小时 / 低风险。不传 tag 时行为与现在完全一致。

---

## 🟢 D. Config dataclass

### 诊断
入口脚本顶部一堆 `ImageSize = 224` `NumClasses = 10` `LearningRate = 0.0001` 大写常量，每次想换超参就要改源码（git 比对模糊、jupyter notebook 之间复制粘贴丢同步）。

### 改法
入口脚本顶部那批常量统一收到一个 PascalCase dataclass：

```python
from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    ImageSizeW   : int   = 224
    ImageSizeH   : int   = 224
    NumClasses   : int   = 10
    LearningRate : float = 0.0001
    BatchSize    : int   = 128

Config = TrainConfig()

# 允许 --LearningRate=3e-4 覆盖（Executor 已经收到 KVArgsForML）
def ApplyConfigFromKV(inConfig, inKVArgs):
    for K, V in inKVArgs.items():
        if hasattr(inConfig, K):
            T = type(getattr(inConfig, K))
            setattr(inConfig, K, T(V))
```

把 `ApplyConfigFromKV` 放 `KongMing/Utils/ConfigUtils.py`。

### 影响文件
- 新建 `KongMing/Utils/ConfigUtils.py`（+20 行）
- 9 个入口脚本各 +5 行（可选改造，不改也没关系）

### 估时 / 风险
工具 30 分钟，9 个入口适配 1.5 小时 / 低风险。可选：先只改一个入口验证。

---

## 🟡 E. NNModuleDict 自动注册

### 诊断
新增网络要在 3 处同步登记：`Trainer.__init__` 的 `self.X = X.to(device)`、`Archiver.NNModuleDict["X"] = X`、`ModelFactory.__init__` 的 `RegisterMultiNNModule`。漏一处就静默丢权重。

### 改法（两个方案）

**方案 1：反射**（省 boilerplate，但失去显式声明）
`BaseModelFactory.__init__` 末尾扫描 `Trainer` 上所有 `BaseNNModel` 实例属性，自动登记到 `Archiver.NNModuleDict`。

**方案 2：装饰器**（保留显式但减少重复）
`@RegisterNN` 装饰器，挂在 Trainer 子类的属性上。

**建议先不做**——当前的"显式三处登记"虽然啰嗦但易读，符合 KongMing 哲学。等真的因为这个踩坑再做。

### 影响文件
- `KongMing/ModelFactory/BaseModelFactory.py`（+20 行反射）

### 估时 / 风险
1 小时 / 中风险（反射可能误抓）。**列在这里仅作为备选**。

---

## 🟢 H. 文件 sentinel fallback（替代 keyboard）

### 诊断
`Executor.Train` 已 try/except 包了 `keyboard.add_hotkey`，Linux/Mac/容器里挂不上时不崩了，但**也没办法 force save / soft exit**。

### 改法
`BaseModelFactory.__BMEndEpochTrain` 开头检查 sentinel 文件：

```python
LogDir = self.Trainer.LogRootPath
SaveSignal = os.path.join(LogDir, ".save")
ExitSignal = os.path.join(LogDir, ".exit")
if os.path.exists(SaveSignal):
    os.remove(SaveSignal)
    self.ForceSaveAtEndEpoch()
if os.path.exists(ExitSignal):
    os.remove(ExitSignal)
    self.ForceExitAtEndEpoch()
```

用户在另一个终端 `touch <log_dir>/.save` 即可触发，无需 root。

### 影响文件
- `KongMing/ModelFactory/BaseModelFactory.py`（+10 行）

### 估时 / 风险
30 分钟 / 零风险。和 keyboard 热键并存，谁先触发都行。

---

## 总结

| 项 | 估时 | 风险 | 收益 |
|---|---|---|---|
| B Delegate 隔离 | 30 min | 零 | 防 callback 静默掉链 |
| A AMP+累积 | 2-3 h | 中 | 显存×2，能跑大网络 |
| C Val 钩子 | 1.5 h | 低 | 看得见过拟合 |
| G jsonl 指标 | 30 min | 零 | 画曲线不用写正则 |
| F ModelTag | 1 h | 低 | 实验对比有维度 |
| D Config dataclass | 2 h | 低 | 超参可 git diff |
| E 自动注册 | 1 h | 中 | 省 boilerplate |
| H 文件 sentinel | 30 min | 零 | 跨平台热键 |

**最小可行包**：B + A + C + G ≈ 半天，把"框架够用度"从 60 分推到 80 分。

剩下的按心情。
