from lmformatenforcer import CharacterLevelParser, StringParser, SequenceParser, UnionParser

def add_string(parser, string):
    for char in string:
        parser = parser.add_character(char)
    return parser

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