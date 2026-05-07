# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目性质

个人 AI 学习项目（Python + PyTorch），用于实现各种经典网络（MathNet、UNet-GAN、Baby GPT、DCGAN、DDPM、PPO、VGG 等）。仓库根目录的 `001_*.py` … `009_*.py` 是按主题编号的可执行入口脚本；`900_*` 系列是零散试验脚本；`jupyter/` 是探索用 notebook。所有可复用代码集中在 `KongMing/` 这个自研小框架下。**没有 README、没有测试套件、没有 CI**——除 `requirements.txt` 外，所有"约定"都隐含在 `KongMing/` 的代码里。

依赖安装：`pip install -r requirements.txt`（`graphviz` 渲染图形还需另外装系统级 Graphviz 可执行文件；`keyboard` 在 Windows 下注册全局热键通常需要管理员权限）。

## 运行入口约定

每个 `00X_*.py` 入口脚本结构高度雷同：构造一个 `XxxModelFactory`，包进 `Executor`，然后根据 `Executor.IsExistModel()` 与命令行参数决定走训练还是 Eval 分支。运行示例：

```
python 005_DDPM.py                  # 已有模型则 Eval，否则新训练
python 005_DDPM.py new              # 强制重新开始训练
python 005_DDPM.py inc              # 增量训练（接最新 checkpoint）
python 005_DDPM.py eval             # 强制 Eval
python 005_DDPM.py inc -epoch=120 -epochitercount=50
python 005_DDPM.py --somekey=value  # 双横线 = 透传给 ML 层
```

参数解析规则在 `KongMing/Utils/Executor.py:76`（`__GetArgs` / `__AnalyzeArgs`）：
- 单 `-` 前缀 → Executor 自身参数，目前识别 `-epoch=N` 与 `-epochitercount=N`
- 双 `--` 前缀 → 透传给 Trainer/Model 的 `inKVArgsForML`
- 裸单词 → `newtrain|new` / `inctrain|inc` / `eval`

训练时 `Executor` 还会绑定全局热键：`Ctrl+S` 在当前 epoch 末强制保存，`Ctrl+X` 在当前 epoch 末软退出（依赖 `keyboard` 包，Windows 下通常需要管理员权限才能注册全局热键——这是一个常见踩坑点）。

### 入口脚本 → 数据集 / 输出目录速查

| 入口 | 数据集 | 备注 |
|---|---|---|
| `001_MathNet.py` | 脚本内合成（`Calculator`，加减乘除幂） | 不依赖 `DatasetPath` |
| `002_UNetGAN.py` | `bFashionMNIST=True` 时用 FashionMNIST，否则 `cartoon_faces/` | torchvision 自动下载 / `DatasetPath` 子目录 |
| `003_baby_gpt.py` | 脚本内合成（0/1 序列） | 不依赖 `DatasetPath`；需 `graphviz` 画图 |
| `004_DCGAN.py` | 同 002 | 同上 |
| `005_DDPM.py` | 同 002 | 同上 |
| `006_PPO.py` | `gym` 环境 `CartPole-v1` | 不依赖 `DatasetPath` |
| `007_VGG16.py` / `008_VGGMNN16.py` / `009_VGG16Mix.py` | CIFAR10 | torchvision 自动下载到 `DatasetPath` |

输出统一落在 `output/<脚本basename>/<数据集名>/`（VGG 系列固定写 `CIFAR10`，GAN/DDPM 用 `ModelFolderByDataset` 变量）。

## 数据与输出路径约定

数据集根目录由 `KongMing.Utils.DatasetPath.ResolveDatasetPath()` 统一解析，按优先级：
1. 环境变量 `STUDYAI_DATA_DIR`（最推荐——避免改代码）
2. `D:/AI/Datasets/`（作者机器历史路径）
3. `D:/__DevAI__/Datasets/`
4. `./data/Datasets/`（兜底，torchvision 内置数据集会自动下载到此处）

下游会去其中找 `cartoon_faces/` 等子目录，或让 torchvision 自行下载（FashionMNIST/CIFAR10 等）。新增入口脚本直接 `from KongMing.Utils.DatasetPath import ResolveDatasetPath; DatasetPath = ResolveDatasetPath()`，不要再复制旧的 `if os.path.exists("D:/AI/")` 那一段。

每个入口脚本的产物落在 `output/<脚本basename>/<数据集名>/`：
- `ArchivedModels/<时间戳>/<NNName>_<epoch>.pkl` — 由 `BaseArchiver` + `FileManagerWithNum` 管理
- `images/<时间戳>.png` — Eval 时生成的样本

`.gitignore` 已忽略 `data`、`trained_models`、`images`、`output`、`__pycache__`、`.ipynb_checkpoints`。

## KongMing 框架架构

四层职责分离，**任何新模型都要按这四层注册**，否则保存/加载/事件回调会失效：

```
ModelFactory ── 组装入口（持有 Trainer + Archiver）
   ├─ Trainer ── 训练循环 + 优化器 + 损失，通过 Delegate 事件对外暴露生命周期
   └─ Archiver ─ 维护 NNModuleDict，负责按 epoch 保存/加载 .pkl
```

关键文件：
- `KongMing/ModelFactory/BaseModelFactory.py` — 把 Trainer 的 `BeginTrain / EndBatchTrain / EndEpochTrain / EndTrain` 四个事件挂到 Archiver 上；从 `inKVArgs` 读取两个调速参数：`SaveInterval`（默认 10，每多少 epoch 落盘一次）与 `PrintInterval`（默认 1，每多少 batch 打印一次损失，赋给 `Trainer.PrintInterval`）。命令行通过双横线透传，例：`python 005_DDPM.py inc --SaveInterval=5 --PrintInterval=20`。
- `KongMing/Trainer/BaseTrainer.py` — 训练主循环 `__DontOverride__Train`（命名暗示子类不要覆盖）；子类必须实现 `_CreateOptimizer`、`_CreateLossFN`、`_BatchTrain`。`ShouldPrintBatch()` 配合 `PrintInterval` 决定是否打印当前 batch（最后一个 batch 总是打印）。
- **反传 API**：基类提供了 `Backpropagate` 上下文管理器与 `_BackPropagate` 静态方法，但当前所有 Trainer 子类**都没用**这两个 —— 实际惯用法是 `with self.<NNComponent> as X:`（`KongMing/Models/BaseNNModel.py:138` 的 `__enter__/__exit__` 通过组件自带的 `BackPropagater` 触发 zero_grad/backward/step，并在退出时自动 `__UpdateEMA`）。组件通过 `ApplyOptimizer / ApplyLossFunc / CalcAndAcceptLoss / AcceptLoss` 配合使用。新写 Trainer 子类请沿用此 `with` 习惯，不要直接调 `optimizer.step()`，否则会跳过 EMA 更新。
- `KongMing/Archiver/BaseArchiver.py` — 通过 `NNModuleDict: {name: nn.Module}` 持有所有要保存的子网；`NNModuleNameOnlyForTrain` 列出只在训练期需要、Eval 前会被 `del` 掉的模块（如 GAN 的 Discriminator）。
- `KongMing/Utils/Delegate.py` — 极简事件总线（`add` / `__call__`），所有跨层通讯都走这个。

`SingleNNModelFactory` vs `MultiNNModelFacotry`（注意源码里的拼写就是 `Facotry`）分别对应单网络与多网络（GAN/UNet+判别器/Diffusion 多组件）。多网络版本要求在子类构造里调用 `Trainer.RegisterMultiNNModule(dict)` 与 `Archiver.RegisterMultiNNModule(dict)`。

## 在新增/修改模型时的注意事项

1. **新模型的位置**：训练逻辑放 `KongMing/Trainer/<DomainName>Trainer.py`；组装放 `KongMing/ModelFactory/<Domain>/<Name>ModelFactory.py`；最后在仓库根加一个 `0XX_<Name>.py` 入口。网络结构按颗粒度拆放：
   - `KongMing/Models/` —— **端到端、可独立训练 / Eval 的整网**，需要继承 `BaseNNModel`（带 `BackPropagater` / EMA / `with` 上下文协议），由 Archiver 注册并落盘。例：`Models/UNets/UNet2DBase.py`。
   - `KongMing/Modules/` —— **可复用的 nn.Module 零件 / block**，是普通 `nn.Module`，不参与持久化、不进 NNModuleDict。例：`Modules/Resampling/{Down,Up}.py`、`Modules/Attentions/`、`Modules/Transformer/`、`Modules/Zoo/{UNet,EMA}.py`、`Modules/PositionEmbedding.py`。判断标准：**这个东西需不需要单独保存权重 / 单独被 Archiver 管理？** 需要 → `Models/`；只是被组合进更大网络 → `Modules/`。
2. **Archiver 的 NNModuleDict 必须填齐**——`IsExistModels()` 通过遍历这个 dict 判断"模型是否已训练过"，漏注册会导致每次启动都重训。
3. **路径里的 timestamp**：`FileManagerWithNum` 每次新训练会建一个新时间戳目录；增量训练靠 `GetValidLatestTimestampDirInfo` 找到"最近的"那个。如果手工删除/移动 `ArchivedModels/` 下的目录，`inc` 模式会找不到旧权重。
4. **依赖**：见 `requirements.txt`（`torch / torchvision / einops / keyboard / gym / graphviz / pandas`）。
5. **设备**：`BaseModelFactory.__init__` 已支持 CPU 回退（CUDA 不可用时打印 "CUDA unavailable, using CPU"），无需再手工绕过。
6. **Executor 对 ModelFactory 的隐式契约**（duck-typed，无 ABC 强制）：实现 `NewTrain / IncTrain / Eval / LoadLastest / IsExistModels / ForceSaveAtEndEpoch / ForceExitAtEndEpoch` 七个方法。`BaseModelFactory` 已全部提供，新增 ModelFactory 子类时**只需实现 Eval**（其它继承基类即可）。
7. **EndEpochTrain 回调注册顺序**：`SingleNNTrainer.__init__` 先于 `BaseModelFactory.__init__`（入口脚本里 Trainer 先 new）→ LR scheduler.step 先跑、Save 后跑。这个顺序是**对的**：Save 失败时 scheduler 已经更新过了。新增回调时把"会失败的 IO"放最后。
8. **DataLoader 必须有 `__len__`**：`BaseTrainer.__DontOverride__EpochTrain` 用 `len(inDataLoader.dataset) / batch_size` 算 `BatchNumPerEpoch`；用 `IterableDataset`（流式）会崩。当前所有入口都是 map-style dataset，没问题。

## 编码风格（项目统一约定，必须沿用）

作者已使用此风格 20 余年，是个人长期偏好，**不是需要"改善"的对象**。新增/修改任何代码都必须沿用：

- 类名、方法名、成员变量、模块内函数：**全部 PascalCase**（不使用 PEP8 的 snake_case）。
- 函数参数以 `in` 前缀（`inDataLoader`、`inKVArgs`、`inEpochIndex`）。
- 布尔变量以 `b` 前缀（`bForceNewTrain`、`bFashionMNIST`）。
- 用名称表达约束的私有方法（例如 `__DontOverride__Train` 表示"不要覆盖此方法"）。
- 入口脚本顶部用大写常量（`EmbeddingDim`、`ImageSize` 等），不使用配置文件。

提改进建议、举代码示例时也要遵守上述风格；不得把"与社区主流不同"作为缺点列出。
