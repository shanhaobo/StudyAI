# 盲区：Windows 下 Python 多进程是 `spawn` 不是 `fork`

**主题大类**：Python / OS（非 AI）

> **典型触发场景**：PyTorch DataLoader 设了 `num_workers>0`，跑出 `RuntimeError: An attempt has been made to start a new process before...` 或死锁。但本质是 Python 多进程行为，不是 PyTorch 特有。

## 误区

`multiprocessing` / `concurrent.futures.ProcessPoolExecutor` / `DataLoader(num_workers=N)` 在 Linux 跑得好好的，搬到 Windows 就崩——以为是 Windows 特别脆。

## 真相

Python 多进程在不同 OS 用不同**启动方式**，决定脚本要怎么写：

| OS | 默认 start method | 含义 |
|---|---|---|
| Linux | `fork` | 子进程直接复制父进程内存，对象状态都带过去 |
| Windows / macOS（新版） | `spawn` | 子进程**重新 import 整个脚本**，从头执行 |

`spawn` 模式下，子进程会再跑一遍你的入口脚本——如果创建子进程的代码写在脚本顶层，子进程一执行就又试图新建子进程，**递归到爆**。

正确写法（Windows 必备）：

```python
if __name__ == "__main__":
    # 任何会启动子进程的代码（DataLoader/Pool/Executor）都必须在这里
    DL = DataLoader(dataset, num_workers=4, ...)
    for batch in DL:
        ...
```

副作用与陷阱：

- `spawn` 会**重新初始化**子进程：模块被重新 import、全局变量重置、随机数种子重置。
- 跨进程传递的对象必须**可 pickle**：lambda、闭包、嵌套类、open 的文件句柄、数据库连接、CUDA tensor 都不行。
- 父进程持有的资源**不会**被子进程继承——所有需要的资源在子进程里**懒加载**（DataLoader 场景：在 `Dataset.__getitem__` 里 open 文件，而不是 `__init__`）。
- 子进程的 `random.seed / np.random.seed / torch.manual_seed` 都从默认值开始——多 worker 想要可预测随机性必须显式注入种子（DataLoader 用 `worker_init_fn`，自家代码自己写）。

## 直观比喻

> `fork` = "克隆一个我，记忆一致"——爸爸怎么样儿子就怎么样。
> `spawn` = "生一个新人，给他剧本让他重演"——所以剧本（脚本顶层）必须有"成年人才执行"的判断（`__main__`）。

## 一句话记住

> **Windows / macOS 下凡是启动子进程的代码必须在 `if __name__ == "__main__":` 里**；子进程不继承资源，需懒加载；随机性靠 `worker_init_fn` 显式注入。
