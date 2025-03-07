from .. import config

from lmformatenforcer import CharacterLevelParser, StringParser, SequenceParser, UnionParser
from enum import Enum
from typing import Optional

def add_string(parser, string):
    for char in string:
        parser = parser.add_character(char)
    return parser

class StateType(str, Enum):
    STRING = 'string'
    START = 'start'
    TAG = 'tag'
    END = 'end'
    INTERMEDIATE_STRING = 'intermediate_string'
    INTERMEDIATE_TAG = 'intermediate_tag'

class OutputType(str, Enum):
    NEW_TAG = 'new_tag'
    TAG = 'tag'
    NONE = 'none'

ParserState = (StateType, str, Optional[str]) # (string|start|tag|end,<rest-to-parse>,<rest-of-tag>)
ParserOutput = (OutputType, Optional[str])

def step(state: ParserState, char: str) -> [(ParserState,ParserOutput)]:
    results = []
    if state[1].startswith(char):
        if state[0] == StateType.STRING:
            results.append(((StateType.STRING,state[1][len(char):],None),(OutputType.NONE,None)))
        elif state[0] == StateType.TAG:
            results.append(((StateType.TAG,state[1][len(char):],None),(OutputType.TAG,char)))
    if state[0] == StateType.TAG and config.TAG_END.startswith(char):
        if config.TAG_END == char:
            results.append(((StateType.STRING,state[1],None),(OutputType.NONE,None)))
        else:
            results.append(((StateType.INTERMEDIATE_STRING,state[1],config.TAG_END[len(char):]),(OutputType.NONE,None)))
    if state[0] == StateType.STRING and config.TAG_START.startswith(char):
        if config.TAG_START == char:
            results.append(((StateType.TAG,state[1],None),(OutputType.NEW_TAG,None)))
        else:
            results.append(((StateType.INTERMEDIATE_TAG,state[1],config.TAG_START[len(char):]),(OutputType.NONE,None)))
    if state[0] == StateType.INTERMEDIATE_STRING:
        if state[2] == char:
            results.append(((StateType.STRING,state[1],None),(OutputType.NONE,None)))
        elif state[2].startswith(char):
            results.append(((StateType.INTERMEDIATE_STRING,state[1],state[2][len(char):]),(OutputType.NONE,None)))
    if state[0] == StateType.INTERMEDIATE_TAG:
        if state[2] == char:
            results.append(((StateType.TAG,state[1],None),(OutputType.NEW_TAG,None)))
        elif state[2].startswith(char):
            results.append(((StateType.INTERMEDIATE_TAG,state[1],state[2][len(char):]),(OutputType.NONE,None)))
    return results
    

def parse(base: str, tagged: str, state: ParserState = None, res = [], idx = 0):
    if state is None:
        state = (StateType.STRING, base, None)
    if len(state[1]) == 0:
        return res
    results = step(state,tagged[0])
    # print(results)
    for result in results:
        cp = []
        cp.extend(res)
        if result[1][0] == OutputType.TAG:
            cp[-1]['tag'] += result[1][1]
        elif result[1][0] == OutputType.NEW_TAG:
            cp.append({
                'tag': '',
                'pos': idx
            })
        new_res = parse(base, tagged[1:], result[0], cp, idx + 1)
        if new_res is not None:
            return new_res
    return None

class NERParser(CharacterLevelParser):
    """A parser that allows an annotated string."""
    def __init__(self, string: str, start: str, end: str, open: bool = True):
        self.string = string
        self.start = start
        self.end = end
        self.open = open
    def add_character(self, new_character: str) -> CharacterLevelParser:
        # print(f"adding '{new_character}'")
        parsers = []
        if self.string.startswith(new_character):
            # print(f"  '{new_character}' matches string part ({self.string})")
            parsers.append(NERParser(self.string[1:], self.start, self.end, self.open))
        if self.open and self.start.startswith(new_character):
            # print(f"  '{new_character}' matches {self.start}")
            parsers.append(SequenceParser([
                StringParser(self.start[1:]),
                NERParser(self.string, self.start, self.end, False)
            ]))
        if (not self.open) and self.end.startswith(new_character):
            # print(f"  '{new_character}' matches {self.end}")
            parsers.append(SequenceParser([
                StringParser(self.end[1:]),
                NERParser(self.string, self.start, self.end, True)
            ]))
        return UnionParser(parsers)
    def get_allowed_characters(self) -> str:
        # print((self.string[0] if self.string else "")+(self.start[0] if self.open else self.end[0]))
        return (self.string[0] if self.string else "")+(self.start[0] if self.open else self.end[0])
    def can_end(self) -> bool:
        return (not self.string) and open