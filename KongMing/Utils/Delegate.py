class Delegate:
    def __init__(self):
        self.functions = []

    def __call__(self, *inArgs, **inKWArgs) -> None:
        for func in self.functions:
            func(*inArgs, **inKWArgs)

    def add(self, func) -> None:
        self.functions.append(func)

    def remove(self, func) -> None:
        self.functions.remove(func)
