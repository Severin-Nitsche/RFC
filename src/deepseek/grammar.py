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
  # We need to escape the tags
  tag_start = "".join(to_unicode_escape(c) for c in tag_start) 
  tag_end = "".join(to_unicode_escape(c) for c in tag_end)

  rules = []
  for i, char in enumerate(text):
    # First, we escape the character if it is a "
    char = to_unicode_escape(char)
    if not i == len(text) - 1:
      rules.append(f'o-{i} ::= "{char}" (o-{i+1} | ("{tag_start}" i-{i+1}))')
      rules.append(f'i-{i} ::= "{char}" (i-{i+1} | ("{tag_end}" o-{i+1}))')
    else:
      rules.append(f'o-{i} ::= "{char}"')
      rules.append(f'i-{i} ::= "{char}" "{tag_end}"')
  rules.append(rf"""root ::= [a-zA-z \"'.,\t\n{tag_start}{tag_end}]* "</think>" (o-0 | ("{tag_start}" i-0))""")
  return "\n".join(rules)
