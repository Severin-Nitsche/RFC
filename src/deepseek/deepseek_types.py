from collections.abc import Mapping
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class PromptType(str, Enum):
    ANNOTATE = 'annotate'
    CLASSIFY = 'classify'
    VERIFY = 'verify'

class ShotType(int, Enum):
    ZERO_SHOT = 0,
    ONE_SHOT = 1,
    FEW_SHOT = 2

@dataclass
class Example(Mapping):
    input: str
    output: str
    entity: Optional[str] = None
    def __getitem__(self, item):
        try:
            return getattr(self, item)
        except:
            raise KeyError(item)
    def __iter__(self):
        raise Exception()
    def __len__(self):
        raise Exception()
    def keys(self):
        return dir(self)

@dataclass 
class PromptInfo(Mapping):
    prompt_type: PromptType
    category: str
    explanation: str
    example: [Example]
    options: Optional[str]
    def __getitem__(self, item):
        try:
            return getattr(self, item)
        except:
            raise KeyError(item)
    def __iter__(self):
        raise Exception()
    def __len__(self):
        raise Exception()
    def keys(self):
        return dir(self)