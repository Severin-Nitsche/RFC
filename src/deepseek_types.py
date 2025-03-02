
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class PromptType(str, Enum):
    ANNOTATE = 'annotate'
    CLASSIFY = 'classify'
    VERIFY = 'verify'

@dataclass
class Example:
    input: str
    output: str
    entity: Optional[str] = None

@dataclass 
class PromptInfo:
    prompt_type: PromptType
    category: str
    explanation: str
    example: [Example]
    options: Optional[str]
    def __getitem__(self, item):
        return getattr(self, item)