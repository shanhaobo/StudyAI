# KongMing 框架 Review（2026-05-07）

完整深度走查后整理的可优化点清单，按严重程度分三档。每条标了"代价"与"状态"。**编码风格相关项（PascalCase / `in/b` 前缀 / `__DontOverride__` 命名）一律不动**，那是项目长期约定。

## 进度总览（2026-05-07）

| | 总条目 | ✅ 已修 | 📝 文档化 | ⏳ 待办 |
|---|---|---|---|---|
| 🔴 第一档 | 4 | 4 (#1, #2, #3, #4) | 0 | 0 |
| 🟡 第二档 | 7 | 5 (#5, #6, #9, #10, #11) | 2 (#7, #8) | 0 |
| 🟢 第三档 | 6 | 4 (#12, #13, #16, #17) | 2 (#14, #15) | 0 |
| **合计** | **17** | **13** | **4** | **0** |

**全部完成**。

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

### 8. 📝 `Delegate` 单一回调失败会中断整条事件链
担忧：`EndEpochTrain` 链上 Save 抛了，scheduler.step 不会被调用。

- **核查结论**：实际注册顺序是 **scheduler.step 先（SingleNNTrainer.__init__），Save 后（BaseModelFactory.__init__）**——因为入口脚本先 new Trainer 再 new Factory。Save 失败时 scheduler 已更新过，安全。
- **状态**：📝 已在 `CLAUDE.md` 第 7 条写明此契约："新增回调时把会失败的 IO 放最后"。无需改 Delegate 代码。

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

## 实施总结

13 条代码改动 + 4 条文档化全部完成。整体约 280 行变动，零外部 API 破坏：

- 旧 `.pkl` 仍可读（legacy 分支）；
- 旧拼写错的类名 `MultiNNModelFacotry` 仍可 import；
- 旧方法名 `GetFilePathAndNameFromTimestampDirPathByEpoch` 仍可调（alias）；
- 旧入口脚本如不改也能跑（手写路径没移除支持）；
- 命令行参数仍接受老用法（`--PrintInterval=5` 之类）。
