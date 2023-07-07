
from typing import Any


class AveragedUtilBase:
    def __init__(self, inInitValue) -> None:
        self.AveragedValue = inInitValue

    def item(self):
        return self.AveragedValue
    

class AveragedUtilBaseWOInitValue(AveragedUtilBase):
    def __init__(self, inAccpetNewValueFunc) -> None:
        super().__init__(None)
        self.AcceptNewValue = self.__AcceptInitValue
        self.__AcceptNewValueFunc = inAccpetNewValueFunc

    def __AcceptInitValue(self, inInitValue):
        self.AveragedValue = inInitValue
        self.AcceptNewValue = self.__AcceptNewValueFunc
        
class EMAValue(AveragedUtilBaseWOInitValue):
    def __init__(self, inDecay) -> None:
        super().__init__(self.__EMAAcceptNewValue)
        self.Decay = inDecay

    def __EMAAcceptNewValue(self, inNewValue):
        self.AveragedValue = (1 - self.Decay) * inNewValue + self.Decay * self.AveragedValue

    def __call__(self, inNewValue) -> Any:
        self.AcceptNewValue(inNewValue)
        return self.AveragedValue
    
