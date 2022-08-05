from grongier.pex import Message
from dataclasses import dataclass

@dataclass()
class MLRequest(Message):
    pass

@dataclass()
class MLResponse(Message):
    output:str = None

