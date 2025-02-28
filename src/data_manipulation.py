import json
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import spacy

class PromptType(Enum):
    ANNOTATE = 'annotate'
    CLASSIFY = 'classify'
    VERIFY = 'verify'

@dataclass
class Example:
    input: str
    output: str
    entity: Optional[str] = None

nlp = spacy.load("en_core_web_sm")

def _format_echr_annotate(data, sent, annotator, category):
    """
    This function extracts all category entities from the annotation and provides the @@-## labelled output
    """
    start_idx, end_idx = sent.start_char, sent.end_char
    entities = list(map(lambda mention:
        {
            'start_offset': mention['start_offset'] - start_idx,
            'end_offset': mention['end_offset'] - start_idx
        },
        filter(lambda mention: 
            mention['entity_type'] == category and
            mention['start_offset'] >= start_idx and
            mention['end_offset'] <= end_idx,
            data['annotations'][annotator]['entity_mentions']
        )
    ))
    entities.sort(key=lambda mention: mention['start_offset'])
    text = sent.text
    result = ''
    r_idx = 0
    for entity in entities:
        result += text[r_idx:entity['start_offset']]
        result += '@@' # TODO: change these hardcoded values
        result += text[entity['start_offset']:entity['end_offset']]
        result += '##'
        r_idx = entity['end_offset']
    result += text[r_idx:]
    return Example(input=text, output=result)

def _format_echr_verify(data, sent, annotator, category):
    """
    This function extracts all category entities from the annotation and constructs positive and negative examples
    """
    start_idx, end_idx = sent.start_char, sent.end_char
    text = sent.text
    return list(map(lambda mention:
        Example(
            input=text,
            output='yes' if mention['entity_type'] == category else 'no',
            entity=text[mention['start_offset'] - start_idx:mention['end_offset'] - start_idx]
        ),
        filter(lambda mention: 
            mention['start_offset'] >= start_idx and
            mention['end_offset'] <= end_idx,
            data['annotations'][annotator]['entity_mentions']
        )
    ))

def _format_echr_classify(data, sent, annotator, category):
    """
    This function extracts all category entities from the annotation and constructs category examples
    """
    start_idx, end_idx = sent.start_char, sent.end_char
    text = sent.text
    return list(map(lambda mention:
        Example(
            input=text,
            output=mention[category],
            entity=text[mention['start_offset'] - start_idx:mention['end_offset'] - start_idx]
        ),
        filter(lambda mention: 
            mention['start_offset'] >= start_idx and
            mention['end_offset'] <= end_idx,
            data['annotations'][annotator]['entity_mentions']
        )
    ))

def process_echr(path, prompt_type: PromptType, category):
    """
    This function converts the echr files to usable examples with the deepseek templates.
    """
    if not prompt_type in PromptType:
        raise ValueError(f"Expected PromptType, got '{prompt_type}'")
    with open(path, 'r', encoding='utf-8') as file:
        raw = json.load(file)
        flatten = []

        for data in raw:
            doc = nlp(data['text'])
            for sent in doc.sents:
                for annotator in data['annotations']:
                    if prompt_type == PromptType.ANNOTATE:
                        flatten.append(_format_echr_annotate(data, sent, annotator, category))
                    elif prompt_type == PromptType.CLASSIFY:
                        flatten.extend(_format_echr_classify(data, sent, annotator, category))
                    elif prompt_type == PromptType.VERIFY:
                        flatten.extend(_format_echr_verify(data, sent, annotator, category))
        return flatten