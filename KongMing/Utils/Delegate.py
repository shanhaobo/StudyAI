class Delegate:
    def __init__(self):
        self.FunctionList = []

    def __call__(self, *inArgs, **inKVArgs) -> None:
        for tFunc in self.FunctionList:
            tFunc(*inArgs, **inKVArgs)

    def add(self, func) -> None:
        self.FunctionList.append(func)

    def remove(self, func) -> None:
        self.FunctionList.remove(func)
