from grongier.pex import Message
from dataclasses import dataclass

@dataclass()
class TrainRequest(Message):
    pass

@dataclass()
class MLResponse(Message):
    output:str = None

