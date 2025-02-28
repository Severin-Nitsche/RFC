import config

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

entity_types = ['PERSON', 'CODE', 'LOC', 'ORG', 'DEM', 'DATETIME', 'QUANTITY', 'MISC']
entity_type_explanations = {
    'PERSON': 'Names of people, including nicknames/aliases, usernames and initials',
    'CODE': 'Numbers and codes that identify something, such as SSN, phone number, passport number, license plate',
    'LOC': 'Places and locations, such as: Cities, areas, countries, etc.; Addresses; Named infrastructures (bus stops, bridges, etc.)',
    'ORG': 'Names of organisations, such as: public and private companies; schools, universities, public institutions, prisons, healthcare institutions non-governmental organisations, churches, etc.',
    'DEM': 'Demographic attribute of a person, such as: Native language, descent, heritage, ethnicity; Job titles, ranks, education; Physical descriptions, diagnosis, birthmarks, ages',
    'DATETIME': 'Description of a specific date (e.g. October 3, 2018), time (e.g. 9:48 AM) or duration (e.g. 18 years).',
    'QUANTITY': 'Description of a meaninful quantity, e.g. percentages or monetary values.',
    'MISC': 'Every other type of information that describes an individual and that does not belong to the categories person, codes, location, organization, demographic, datetime or quantity.'
}
identifier_types = ['DIRECT_ID', 'QUASI_ID', 'NO_MASK']
identifier_type_explanations = {
    'DIRECT_ID': 'text spans that directly and unequivocally identify the individual to protect) in the case and should therefore be masked.',
    'QUASI_ID': 'text spans that should be masked since they may lead to the re-identification of the individual when combined with other (not masked) quasi-identifiers mentioned in the text along with public background knowledge.',
    'NO_MASK': 'entities that are neither direct- nor quasi-identifiers, and should therefore not need to be masked. Most entities will belong to this category.'
}
confidential_statuses = ['BELIEF', 'POLITICS', 'SEX', 'ETHNIC', 'HEALTH', 'NOT_CONFIDENTIAL']
confidential_status_explanations = {
    'BELIEF': 'Religious or philosophical beliefs',
    'POLITICS': 'Political opinions, trade union membership',
    'SEX': 'Sexual orientation or sex life',
    'ETHNIC': 'Racial or ethnic origin',
    'HEALTH': 'Health, genetic and biometric data. This includes sensitive health-related habits, such as substance abuse',
    'NOT_CONFIDENTIAL': 'Not confidential information (most entities)'
}

entity_examples = dict(map(lambda entity_type: process_echr(config.ECHR_DEV, PromptType.ANNOTATE, entity_type), entity_types))
entity_verify_examples = dict(map(lambda entity_type: process_echr(config.ECHR_DEV, PromptType.VERIFY, entity_type), entity_types))
indentifier_type_examples = process_echr(config.ECHR_DEV, PromptType.CLASSIFY, 'identifier_type')
confidential_status_examples = process_echr(config.ECHR_DEV, PromptType.CLASSIFY, 'confidential_status')