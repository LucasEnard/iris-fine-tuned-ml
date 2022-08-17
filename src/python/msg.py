from grongier.pex import Message
from dataclasses import dataclass

@dataclass()
class TrainRequest(Message):
    pass

@dataclass()
class TrainResponse(Message):
    pass

@dataclass()
class OverrideRequest(Message):
    pass

@dataclass()
class OverrideResponse(Message):
    text:str = None

@dataclass()
class MLRequest(Message):
    pass

@dataclass()
class MLResponse(Message):
    pass

