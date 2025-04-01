import re

def _construct_think_set(text: str, *additional):
  def char_set(low, high):
    return {chr(c) for c in range(ord(low), ord(high) + 1)}

  base = char_set('a', 'z') | char_set('A', 'Z') | char_set('0', '9') | {'.', '"', "'", ' ', '\n', '\t', ','}
  allowed = set(text)
  for addition in additional:
    allowed = allowed | set(addition)
  special = allowed - base

  return "".join(special)

def _escape_char(char):
  return rf'\U{ord(char):08X}'

def _escape(word):
  return "".join(_escape_char(c) for c in word)

def _think_grammar(*set_params):
  return rf"""[a-zA-Z0-9 \"'.,\t\n{_escape(_construct_think_set(*set_params))}]* "</think>" [\n]* """

def construct_choice_grammar(text, choices):
  choice_grammar = ' | '.join(map(lambda choice: f'"{_escape(choice)}"', choices))
  return rf'root ::= ({choice_grammar})'

def construct_grammar(text: str, tag_start: str, tag_end: str):
  """
  This function creates a GBNR grammar for the given text, that can be used for structured decoding.
  text: The text to reproduce
  tag_start: The opening tag
  tag_end: The closing tag

  Example:
  text: The quick brown fox jumps over the lazy dog.
  tag_start: @@
  tag_end: ##
  -> The resulting grammar will allow arbitrary tagging on the text, for example:
    The @@quick## @@brown## fox jumps over the @@lazy## dog.
  """
  think_regex = _think_grammar(text, tag_start, tag_end)
  # We need to escape the tags
  tag_start = _escape(tag_start)
  tag_end = _escape(tag_end)

  rules = []
  groups = re.split(r'(\s+)', text)
  for i, word in enumerate(groups):
    # First, we escape the character if it is a "
    word = _escape(word)
    if not i == len(groups) - 1:
      rules.append(f'o-{i} ::= "{word}" (o-{i+1} | ("{tag_start}" i-{i+1}))')
      rules.append(f'i-{i} ::= "{word}" (i-{i+1} | ("{tag_end}" o-{i+1}))')
    else:
      rules.append(f'o-{i} ::= "{word}"')
      rules.append(f'i-{i} ::= "{word}" "{tag_end}"')
  rules.append(rf"""root ::= (o-0 | ("{tag_start}" i-0))""")
  return "\n".join(rules)
