import config

import random
import json
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import spacy
import tqdm

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

def preprocess_echr(path):
    with open(path, 'r', encoding='utf-8') as file:
        raw = json.load(file)
        for data in tqdm.tqdm(raw, desc='Preprocessing echr'):
            data['nlp'] = nlp(data['text'])
        return raw      

def process_echr(echr, prompt_type: PromptType, category):
    """
    This function converts the echr files to usable examples with the deepseek templates.
    """
    if not prompt_type in PromptType:
        raise ValueError(f"Expected PromptType, got '{prompt_type}'")

    flatten = []

    def _process(func):
        for data in echr:
            doc = data['nlp']
            for sent in doc.sents:
                for annotator in data['annotations']:
                    func(data, sent, annotator, category)
    
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

    if prompt_type == PromptType.ANNOTATE:
        _process(lambda data, sent, annotator, category: flatten.append(_format_echr_annotate(data, sent, annotator, category)))
    elif prompt_type == PromptType.CLASSIFY:
        _process(lambda data, sent, annotator, category: flatten.extend(_format_echr_classify(data, sent, annotator, category)))
    elif prompt_type == PromptType.VERIFY:
        _process(lambda data, sent, annotator, category: flatten.extend(_format_echr_verify(data, sent, annotator, category)))

    return flatten

nlp = spacy.load("en_core_web_sm")

entity_types = ['PERSON', 'CODE', 'LOC', 'ORG', 'DEM', 'DATETIME', 'QUANTITY', 'MISC']
identifier_types = ['DIRECT_ID', 'QUASI_ID', 'NO_MASK']
confidential_statuses = ['BELIEF', 'POLITICS', 'SEX', 'ETHNIC', 'HEALTH', 'NOT_CONFIDENTIAL']

explanations = {
    'PERSON': 'Names of people, including nicknames/aliases, usernames and initials',
    'CODE': 'Numbers and codes that identify something, such as SSN, phone number, passport number, license plate',
    'LOC': 'Places and locations, such as: Cities, areas, countries, etc.; Addresses; Named infrastructures (bus stops, bridges, etc.)',
    'ORG': 'Names of organisations, such as: public and private companies; schools, universities, public institutions, prisons, healthcare institutions non-governmental organisations, churches, etc.',
    'DEM': 'Demographic attribute of a person, such as: Native language, descent, heritage, ethnicity; Job titles, ranks, education; Physical descriptions, diagnosis, birthmarks, ages',
    'DATETIME': 'Description of a specific date (e.g. October 3, 2018), time (e.g. 9:48 AM) or duration (e.g. 18 years).',
    'QUANTITY': 'Description of a meaninful quantity, e.g. percentages or monetary values.',
    'MISC': 'Every other type of information that describes an individual and that does not belong to the categories person, codes, location, organization, demographic, datetime or quantity.',
    'identifier_type': 'DIRECT_ID are text spans that directly and unequivocally identify the individual to protect) in the case and should therefore be masked.\nQUASI_ID are text spans that should be masked since they may lead to the re-identification of the individual when combined with other (not masked) quasi-identifiers mentioned in the text along with public background knowledge.\nNO_MASK are entities that are neither direct- nor quasi-identifiers, and should therefore not need to be masked. Most entities will belong to this category.',
    'confidential_status': 'BELIEF refers to Religious or philosophical beliefs\nPOLITICS refers to Political opinions, trade union membership\nSEX refers to Sexual orientation or sex life\nETHNIC refers to Racial or ethnic origin\nHEALTH refers to Health, genetic and biometric data. This includes sensitive health-related habits, such as substance abuse\nNOT_CONFIDENTIAL refers to Not confidential information (most entities)'
}

echr = preprocess_echr(config.ECHR_DEV)

examples = {
    'annotate': dict(map(lambda entity_type: (entity_type, process_echr(echr, PromptType.ANNOTATE, entity_type)), entity_types)),
    'verify': dict(map(lambda entity_type: (entity_type, process_echr(echr, PromptType.VERIFY, entity_type)), entity_types)),
    'classify': {
        'identifier_type': process_echr(echr, PromptType.CLASSIFY, 'identifier_type'),
        'confidential_status': process_echr(echr, PromptType.CLASSIFY, 'confidential_status')
    }
}

def get_info(category: str, prompt_type: PromptType, example_is_undesired, options: Optional[str]=None, examples: dict=examples, explanation: dict=explanations, num_examples: int=3, max_try: int=42):
    info = PromptInfo(prompt_type, category, explanation[category], [None]*num_examples, options)
    for i in range(num_examples):
        for _ in range(max_try):
            example = random.choice(examples[prompt_type][category])
            info.example[i] = example # We will *always* have an example, but it might be bad
            if not example_is_undesired(example, i):
                break
    return info

confidential_status_default = lambda example, _: example.output == 'NOT_CONFIDENTIAL'
identifier_type_default = lambda example, _: example.output == 'NO_MASK'
verify_default = lambda example, i: (example.output == 'no' or i % 2 == 1) and (example.output == 'yes' or i % 2 == 0)
annotate_default = lambda example, _: example.output == example.input

optionify = lambda options: ', '.join(options)

# print('====== Classify =======')
# print(get_info('confidential_status',PromptType.CLASSIFY,confidential_status_default,optionify(confidential_statuses)))
# print('====== Annotate =======')
# print(get_info('PERSON',PromptType.ANNOTATE,annotate_default))
# print('====== Verify =======')
# print(get_info('CODE',PromptType.VERIFY,verify_default))