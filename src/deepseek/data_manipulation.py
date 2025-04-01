from .. import config

from .deepseek_types import PromptType, Example, PromptInfo

import json
import spacy
import tqdm

nlp = spacy.load('en_core_web_sm')

def preprocess(path, accessor, desc):
  with open(path, 'r') as file:
    raw = json.load(file)
    for data in tqdm.tqdm(raw, desc=f'Preprocessing {desc}'):
      data['nlp'] = nlp(accessor(data))
    return raw

def annotate(text, entities):
  entities = sorted(entities, key=lambda mention: mention['start_offset'])
  result = []
  r_idx = 0
  for entity in entities:
    result.append(text[r_idx:entity['start_offset']])
    result.append(config.TAG_START)
    result.append(text[entity['start_offset']:entity['end_offset']])
    result.append(config.TAG_END)
    r_idx = entity['end_offset']
  result.append(text[r_idx:])
  return ''.join(result)

def process_echr(echr, prompt_type: PromptType, category):
  """
  echr: The echr dataset enriched with nlp information.
    Shape: [{
      annotations: {
        <annotator>: {
          entity_mentions: [{
            start_offset: <start_offset>
            end_offset: <end_offset>
            entity_type: <entity_type>
            confidential_status: <confidential_status>
            identifier_type: <identifier_type>
          }]
        }
      }
      text: <text>
      nlp: <nlp>
    }]
  prompt_type; category: The type of prompt to construct examples for; Determines what `category` refers to
    Values:
      - ANNOTATE > `category` refers to `entity_type`
      - CLASSIFY > `category` refers to either of the keys `confidential_status` or `identifier_type`
      - VERIFY > `category` refers to `entity_type`
  
  returns a list of Examples which are ONE sentence each
  """
  examples = []
  for data in echr:
    for annotator in data['annotations']:
      # At this point, we filter the annotations regarding category
      # This will reduce the space for `prompt_type` ANNOTATE
      annotations_ = data['annotations'][annotator]['entity_mentions']\
        if not prompt_type == PromptType.ANNOTATE else\
          filter(
            lambda entity_mention:\
              entity_mention['entity_type'] == category,
            data['annotations'][annotator]['entity_mentions']
          )
      # Save the annotations in a list bc the next filter kills them otherwise
      annotations_ = list(map(dict, annotations_))
      for sent in data['nlp'].doc.sents:
        # At this point we filter the annotations regarding offsets
        annotations = filter(
          lambda entity_mention:\
            entity_mention['start_offset'] >= sent.start_char and
            entity_mention['end_offset'] <= sent.end_char,
          annotations_
        )
        # Save the annotations - the filter iterator is a bastard
        annotations = list(annotations)
        # Now, we map the remaining offsets to fit to the sentence
        # We can safely alter the original element
        #   because it appears in only one sentence
        for entity_mention in annotations:
          entity_mention['start_offset'] -= sent.start_char
          entity_mention['end_offset'] -= sent.start_char
          # We may want to keep the original information
          # entity_mention['offset'] = sent.start_char
        # At this point we differentiate between the prompt_type
        # For ANNOTATE, every *annotated* sentence is an example;
        # For VERIFY, CLASSIFY every entity in a sentence is an example
        if prompt_type == PromptType.ANNOTATE:
          if any(annotations):
            examples.append(Example(
              input=sent.text.strip(), 
              output=annotate(sent.text, annotations).strip()
            ))
        else:
          for entity_mention in annotations:
            if prompt_type == PromptType.VERIFY:
              output = 'yes' if entity_mention['entity_type'] == category\
                else 'no'
            elif prompt_type == PromptType.CLASSIFY:
              output = entity_mention[category]
            examples.append(Example(
              input=annotate(sent.text, [entity_mention]).strip(),
              output=output,
              entity=sent.text[
                entity_mention['start_offset']:entity_mention['end_offset']
              ]
            ))
  return examples
