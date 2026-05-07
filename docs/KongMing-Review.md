# KongMing 框架 Review（2026-05-07）

完整深度走查后整理的可优化点清单，按严重程度分三档。每条标了"代价"与"状态"。**编码风格相关项（PascalCase / `in/b` 前缀 / `__DontOverride__` 命名）一律不动**，那是项目长期约定。

## 进度总览（2026-05-07 全部完成 + Roadmap 全部完成）

| | 总条目 | ✅ 已修 | 📝 文档化 | ⏳ 待办 |
|---|---|---|---|---|
| 🔴 第一档 | 4 | 4 (#1, #2, #3, #4) | 0 | 0 |
| 🟡 第二档 | 7 | 6 (#5, #6, **#8**, #9, #10, #11) | 1 (#7) | 0 |
| 🟢 第三档 | 6 | 5 (#12, #13, #15, #16, #17) | 1 (#14) | 0 |
| **合计** | **17** | **15** | **2** | **0** |

> 自首次 Review 以来 **#8** 从"📝 文档化"升级为"✅ 已修"——Roadmap B 项落地了 `Delegate.bIsolateFailure`（`9f7500a`），不再仅靠注册顺序约定保证。
>
> Review 之外另起的 Roadmap（A-H 8 项）也已全部完成，详见 [`KongMing-Roadmap.md`](./KongMing-Roadmap.md)。

### 与 Roadmap 的对照速查

| Review 条目 | Roadmap 条目 | 关系 |
|---|---|---|
| #8 Delegate 单回调失败 | B Delegate 失败隔离 | Roadmap 把 #8 从纯文档化升级为代码改动 |
| #14 `__len__` 假设 | — | 仍仅文档化（IterableDataset 用例没出现） |
| 其它 #1-#17 | — | Review 闭环；Roadmap 是新一轮针对架构异味的扩展（AMP / Val / jsonl / ModelTag / sentinel / Config / 自动注册） |

---

## 🔴 第一档：会影响训练质量 / 真实 bug

### 1. ✅ 增量训练丢优化器状态
`Archiver.Save/Load` 只存 `Model.state_dict()`，不存 `optimizer / lr_scheduler`。`inc` 重启时 Adam 的 `m, v` 一阶/二阶矩归零，损失抖动。

- **改法**：`Save` 时保存 `{"Model": state_dict, "BackPropagater": {"Optimizer": ..., "LRScheduler": ...}}`；优化器创建滞后于 Load → `BackPropagater._PendingState` 缓存，`ApplyOptimizer / ApplyLRScheduler` 时再灌入。
- **状态**：✅ 已实施。旧 `.pkl`（裸 OrderedDict）legacy 分支可读。

### 2. ✅ `BaseNNModel.EMAHolder / EMATargeModule` 是类属性（全局单例）
所有 `BaseNNModel` 实例共享一份 EMA。GAN 等多网络场景下 G 和 D 都 `ApplyEMA` 会互相覆盖。

- **改法**：改为类级 `_EMARegistry: dict[id(target)→EMAModle]`，每 target 独立。保留 `self.EMA` 子模块语义不破坏旧 state_dict 形态。直接把 target 放成实例属性会被 `nn.Module.__setattr__` 当作子模块注册，造成 `state_dict()` 自指递归——这是落地时多走的一步。
- **状态**：✅ 已实施。

### 3. ✅ `BaseArchiver.Eval()` 用 `del` 永久毁掉训练模块
旧实现 `del NNModuleDict[Name]` 让 Eval 后再 inc 丢 D。

- **改法**：换成 `Module.eval() + .cpu() + torch.cuda.empty_cache()`；`IsExistModel` 配套跳过 `NNModuleNameOnlyForTrain` 里的名字（它们没有持久化语义）。
- **状态**：✅ 已实施。

### 4. ✅ `Executor.__GetArgs` 把 value 也 casefold 了
`--ModelTag=Run_A` 会变成 `modeltag=run_a`，路径/tag 大小写丢失。

- **改法**：用 `partition("=")` 拆分，只对 key 做 `casefold()`；用 `lstrip("-")` 替代 `replace("-","")` 防止 key 内含 `-` 时被错误删除。
- **状态**：✅ 已实施。

---

## 🟡 第二档：架构异味 / 易踩坑

### 5. ✅ `BaseArchiver.__init__` 默认参数是可变 list
`inNNModuleNameOnlyForTrain : list = []` 经典 mutable-default 反模式。同样问题在 `MultiNNArchiver` 与 `MultiNNModelFacotry`。

- **状态**：✅ 已实施（默认 None + 内部赋默认）。

### 6. ✅ `keyboard` 注册是硬依赖
Linux 非 root / Mac / 容器里 `keyboard.add_hotkey` 抛权限错误，整个训练跑不起来。

- **改法**：包 `try/except Exception` 并 print warning，训练继续。
- **状态**：✅ 已实施。

### 7. 📝 `Trainer` 与 `ModelFactory` 的隐式接口未声明
`Executor` 要求传入对象有 `NewTrain/IncTrain/Eval/LoadLastest/IsExistModels/ForceSaveAtEndEpoch/ForceExitAtEndEpoch` 七个方法。

- **状态**：📝 文档化到 `CLAUDE.md` 第 6 条。未引入 `typing.Protocol`（按"框架不膨胀"原则）。

### 8. ✅ `Delegate` 单一回调失败会中断整条事件链
担忧：`EndEpochTrain` 链上 Save 抛了，scheduler.step 不会被调用。

- **首轮核查**：实际注册顺序是 **scheduler.step 先（SingleNNTrainer.__init__），Save 后（BaseModelFactory.__init__）**——因为入口脚本先 new Trainer 再 new Factory。Save 失败时 scheduler 已更新过，所以 Review 阶段评估为"靠注册顺序保证、无需改 Delegate"。
- **后续升级（Roadmap B / `9f7500a`）**：把"靠注册顺序"这条隐契约升级为代码层面强制——`Delegate(bIsolateFailure=True)` 在 `EndEpochTrain` / `EndTrain` 上启用，遍历所有 callback 收集异常并末尾聚合 raise；新增 callback 不再依赖记忆"会失败的 IO 放最后"。
- **状态**：✅ 已实施。`CLAUDE.md` 第 7 条"注册顺序"契约依然有参考价值（描述 scheduler 与 Save 的实际顺序），但不再是唯一依靠。

### 9. ✅ `BaseFileManager.GetFilePathAndNameFromTimestampDirPathByEpoch` 名字误导
名字暗示"最近的有效时间戳目录"，实际是**遍历所有时间戳目录**找该 epoch。

- **改法**：重命名为 `FindFileAcrossAllTimestampsByEpoch`（与 `_Root` 同步），加 docstring 说明语义；旧名保留为 alias 不破坏外部调用。
- **状态**：✅ 已实施。

### 10. ✅ `BaseArchiver._Save` 一次保存多个网络无原子性
循环里逐个 `torch.save`，中间断电会留下"半保存"epoch。

- **改法**：先全部写到 `*.pkl.tmp`，全部成功后批量 `os.replace` 重命名；任一失败清理已写 .tmp。
- **状态**：✅ 已实施。

### 11. ✅ `MultiNNModelFacotry` 拼写
- **改法**：保留旧名作主类名；新增 `MultiNNModelFactory = MultiNNModelFacotry` alias。
- **状态**：✅ 已实施。

---

## 🟢 第三档：可选小优化

### 12. ✅ `print` 没落盘
`Trainer.LogRootPath` 已赋值但没人用。

- **改法**：`BaseModelFactory` 在 `__BMBeginTrain` tee `sys.stdout` 到 `LogRootPath/train.log`，`__BMEndTrain` 还原。每条 print 同时落屏 + 落盘。
- **状态**：✅ 已实施。

### 13. ✅ `EMAValue(0.99)` decay 硬编码
- **改法**：`BaseModelFactory` 在 `__BMBeginTrain` 读 `inKVArgs["LossEMADecay"]`，遍历 `NNModuleDict` 给所有 `BaseNNModel` 的 `BackPropagater._AvgLoss.Decay` 推送。命令行：`--LossEMADecay=0.9`。
- **状态**：✅ 已实施。

### 14. 📝 `BatchNumPerEpoch` 假设 dataset 有 `__len__`
- **状态**：📝 文档化到 `CLAUDE.md` 第 8 条。未来若引入 `IterableDataset` 再改。

### 15. 📝 `BaseModelFactory.IsExistModels()` vs `Executor.IsExistModel()` 命名不一致
- **改法**：在 `BaseModelFactory` 加 `IsExistModel = IsExistModels` alias。
- **状态**：✅ 已实施（实际是改了，状态升级为 ✅）。

### 16. ✅ `torch.load(path)` 在 PyTorch 2.6+ 默认 `weights_only=True`
- **改法**：所有 `torch.load` 调用显式传 `weights_only=True`。我们存的内容是 dict / OrderedDict / Tensor / Python 标量纯白名单，安全。
- **状态**：✅ 已实施。

### 17. ✅ 入口脚本里 `output/<basename>/<dataset>/` 是手写的
`KongMing/Utils/OutputPath.py` 提供 `BuildOutputPath(__file__, datasetName)` helper。

- **改法**：helper 用 `os.path.join` + `os.makedirs(exist_ok=True)`，签名要求显式传 `__file__`（不从 sys.argv[0] 推导，因为 jupyter / pytest / 嵌套调用会指错）。
- **影响范围**：6 个入口脚本（002, 004, 005, 007, 008, 009）；001/003/006 不用框架（合成数据 / GPT toy / Gym），无 output 概念。
- **特殊处理**：009 的跨脚本加载（`output/008_VGGMNN16/CIFAR10`）改成 `BuildOutputPath("008_VGGMNN16", "CIFAR10")`——helper 的 basename 提取对裸字符串和带 `.py` 的都 work。
- **状态**：✅ 已实施。

---

## 实施总结（含 Roadmap 后续）

**Review 17 项**：15 条代码改动 + 2 条文档化全部完成。整体约 280 行变动，零外部 API 破坏：

- 旧 `.pkl` 仍可读（legacy 分支）；
- 旧拼写错的类名 `MultiNNModelFacotry` 仍可 import；
- 旧方法名 `GetFilePathAndNameFromTimestampDirPathByEpoch` 仍可调（alias）；
- 旧入口脚本如不改也能跑（手写路径没移除支持）；
- 命令行参数仍接受老用法（`--PrintInterval=5` 之类）。

**Roadmap 8 项后续**：B/A/C/G/F/H/D/E 全部完成，详见 [`KongMing-Roadmap.md`](./KongMing-Roadmap.md)。增量约 350 行代码 + 9 个入口脚本套 dataclass，仍保持零 API 破坏（不传任何新 CLI 参数行为完全一致）。

| 阶段 | 条目数 | 关键 commit 起点 → 终点 |
|---|---|---|
| Review (17 项) | 15 ✅ + 2 📝 | `1cfd132` → `4d497d1` |
| Roadmap (8 项) | 8 ✅ | `9f7500a` → `26c7caa` |
| Roadmap 自身文档 | 1 | `21322d6` (初版) → `ff98614` (标记完成) |

合计有效改动 ≈ 630 行，覆盖 11 个核心模块 + 9 个入口脚本，无任何下游被破坏。
