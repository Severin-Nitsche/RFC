import re

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
  # Convert characters to Unicode escape sequences
  def to_unicode_escape(char):
    return rf'\U{ord(char):08X}'
  def word_to_escapes(word):
    return "".join(to_unicode_escape(c) for c in word)
  def char_set(low, high):
    return {chr(c) for c in range(ord(low), ord(high) + 1)}

  default_set = char_set('a', 'z') | char_set('A', 'Z') | char_set('0', '9') | {'.', '"', "'", ' ', '\n', '\t', ','}
  text_set = set(text)
  start_set = set(tag_start)
  end_set = set(tag_end)
  additional = word_to_escapes("".join((text_set | start_set | end_set) - default_set))

  # We need to escape the tags
  tag_start = word_to_escapes(tag_start)
  tag_end = word_to_escapes(tag_end)

  rules = []
  groups = re.split(r'(\s+)', text)
  for i, word in enumerate(groups):
    # First, we escape the character if it is a "
    word = word_to_escapes(word)
    if not i == len(groups) - 1:
      rules.append(f'o-{i} ::= "{word}" (o-{i+1} | ("{tag_start}" i-{i+1}))')
      rules.append(f'i-{i} ::= "{word}" (i-{i+1} | ("{tag_end}" o-{i+1}))')
    else:
      rules.append(f'o-{i} ::= "{word}"')
      rules.append(f'i-{i} ::= "{word}" "{tag_end}"')
  rules.append(rf"""root ::= [a-zA-z0-9 \"'.,\t\n{additional}]{{0,{4096-len(text)*2}}} "</think>" (o-0 | ("{tag_start}" i-0))""")
  return "\n".join(rules)
