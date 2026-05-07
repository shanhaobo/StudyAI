class Delegate:
    """极简事件总线。
    - 默认严格模式：任一 callback 抛异常会直接中断整条链（早 fail 早暴露）。
    - bIsolateFailure=True：遍历所有 callback 收集异常，结尾汇总 raise。
      用于"事件链上有 IO 类 callback（Save / log close）"的场景，避免一个 callback
      失败把后续 scheduler.step 之类的关键工作也吞掉。
    """
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
                Name = getattr(tFunc, "__qualname__", getattr(tFunc, "__name__", repr(tFunc)))
                print("[Delegate] callback {} raised: {}".format(Name, e))
                Errors.append((Name, e))

        if Errors:
            Names = ", ".join(N for N, _ in Errors)
            raise RuntimeError("Delegate had {} callback failure(s): {}".format(len(Errors), Names))

    def add(self, func) -> None:
        self.FunctionList.append(func)

    def remove(self, func) -> None:
        self.FunctionList.remove(func)
