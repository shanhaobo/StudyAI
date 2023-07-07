class Delegate:
    def __init__(self):
        self.functions = []

    def __call__(self, inArgs, inKVArgs) -> None:
        for func in self.functions:
            func(inArgs, inKVArgs)

    def add(self, func) -> None:
        self.functions.append(func)

    def remove(self, func) -> None:
        self.functions.remove(func)
