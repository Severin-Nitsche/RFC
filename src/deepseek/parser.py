from .. import config

import re

def pre_parse(text):
  match = re.search(r".*</think>[\n]*", text, re.DOTALL)
  if match:
    return text[match.end():]
  else:
    print("Thinking incomplete...")
    return None

def parse(tagged: str, base: str, tag_start: str = config.TAG_START, tag_end: str = config.TAG_END):
  """
  This function extracts all tags from the tagged string.
  Example:
    - base: The quick brown fox jumps over the lazy dog.
    - tagged: The quick @@brown## fox jumps over the lazy dog.
    -> {
      tag: brown,
      pos: 10
    }
  """
  # First, we remove the think part
  # tagged = pre_parse(tagged)
  # if tagged is None:
  #   return []

  # Dirty hack
  l = len(base) - len(base.lstrip())
  base = base.strip()
  tagged = tagged.strip()
  base = (' ' * l) + base
  tagged = (' ' * l) + tagged
  # We do not need the tag strings
  # But it is easier if we have them
  # We assume flat tagging because we force the generation this way
  # Therefore, we just need to identify, where the strings do not match
  # Then everything in between is a tag
  tags = []
  tag = False
  tagged_idx = 0
  for base_idx, base_char in enumerate(base):
    try:
      tagged_char = tagged[tagged_idx]
    except:
      print(tagged_idx, base_idx, base_char)
      print(tagged,base)
      break;
    if tagged_char == base_char:
      if tag:
        tags[-1]['tag'] += tagged_char
    else:
      tag = not tag
      if tag:
        tags.append({
          'tag': base_char,
          'pos': base_idx
        })
        tagged_idx += len(tag_start)
      else:
        tagged_idx += len(tag_end)
    tagged_idx += 1
  return tags
