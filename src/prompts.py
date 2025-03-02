import config

from deepseek_types import PromptType, Example, PromptInfo
from data_manipulation import preprocess, process_echr

from typing import Optional
from string import Template
import random

prompt_template = dict()

with open(config.ANNOTATE, 'r') as annotate_file:
    prompt_template[PromptType.ANNOTATE] = Template(annotate_file.read())

with open(config.CLASSIFY, 'r') as classify_file:
    prompt_template[PromptType.CLASSIFY] = Template(classify_file.read())

with open(config.VERIFY, 'r') as verify_file:
    prompt_template[PromptType.VERIFY] = Template(verify_file.read())

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

examples = {}

def construct_examples(echr):
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

def generate_prompt(category: str, prompt_type: PromptType, prompt_input: str, prompt_entity: str=None):
    if not prompt_type in PromptType:
        raise ValueError(f"Expected PromptType, got '{prompt_type}'")

    def _info_to_dict(info):
        res = dict(
            prompt_type = info.prompt_type,
            category = info.category,
            explanation = info.explanation,
            options = info.options,
            prompt_input = prompt_input,
            prompt_entity = prompt_entity
        )
        for i in range(len(info.example)):
            res[f'example_{i}_input'] = info.example[i].input
            res[f'example_{i}_output'] = info.example[i].output
            res[f'example_{i}_entity'] = info.example[i].entity
        return res

    if prompt_type == PromptType.ANNOTATE:
        return prompt_template[prompt_type].substitute(_info_to_dict(
            get_info(category, prompt_type, annotate_default)
        ))
    elif prompt_type == PromptType.CLASSIFY:
        if category == 'confidential_status':
            return prompt_template[prompt_type].substitute(_info_to_dict(get_info(
                category, 
                prompt_type, 
                confidential_status_default, 
                optionify(confidential_statuses)
            )))
        elif category == 'identifier_type':
            return prompt_template[prompt_type].substitute(_info_to_dict(get_info(
                category, 
                prompt_type, 
                identifier_type_default, 
                optionify(identifier_types)
            )))
        else:
            raise ValueError(f'Unknown category for prompt type {prompt_type}: {category}')
    elif prompt_type == PromptType.VERIFY:
        return prompt_template[prompt_type].substitute(_info_to_dict(
            get_info(category, prompt_type, verify_default, num_examples=2)
        ))

# print('====== Annotate =======')
# print(generate_prompt('PERSON', PromptType.ANNOTATE,'He has moved out on his own but still keeps some contact with his dad, mainly because he wants to wait until Maria leaves before cutting ties completely.'))
# print('====== Classify =======')
# print(generate_prompt('confidential_status', PromptType.CLASSIFY,'He has moved out on his own but still keeps some contact with his dad, mainly because he wants to wait until Maria leaves before cutting ties completely.', 'Maria'))
# print('====== Verify =======')
# print(generate_prompt('CODE', PromptType.VERIFY,'He has moved out on his own but still keeps some contact with his dad, mainly because he wants to wait until Maria leaves before cutting ties completely.','dad'))