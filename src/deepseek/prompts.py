from .. import config

from .deepseek_types import PromptType, Example, PromptInfo, ShotType
from .data_manipulation import process_echr, annotate, preprocess

from typing import Optional
from string import Template
import random
import json

prompt_template = dict()

with open(config.ANNOTATE, 'r') as annotate_file,\
    open(config.ANNOTATE_EXAMPLE, 'r') as annotate_example,\
    open(config.CLASSIFY, 'r') as classify,\
    open(config.CLASSIFY_EXAMPLE, 'r') as classify_example,\
    open(config.VERIFY, 'r') as verify,\
    open(config.VERIFY_EXAMPLE, 'r') as verify_example:
    prompt_template[PromptType.ANNOTATE] = dict(
        base = Template(annotate_file.read()),
        example = Template(annotate_example.read())
    )
    prompt_template[PromptType.CLASSIFY] = dict(
        base = Template(classify.read()),
        example = Template(classify_example.read())
    )
    prompt_template[PromptType.VERIFY] = dict(
        base = Template(verify.read()),
        example = Template(verify_example.read())
    )

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

echr = preprocess(config.ECHR, lambda data: data['text'], 'echr')

if config.SHOTS > 0:
    examples = dict(
        annotate = dict(map(lambda entity_type: (entity_type, process_echr(echr, PromptType.ANNOTATE, entity_type)), entity_types)),
        verify = dict(map(lambda entity_type: (entity_type, process_echr(echr, PromptType.VERIFY, entity_type)), entity_types)),
        classify = {
            'identifier_type': process_echr(echr, PromptType.CLASSIFY, 'identifier_type'),
            'confidential_status': process_echr(echr, PromptType.CLASSIFY, 'confidential_status')
        }
    )
else:
    examples = dict()

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

def generate_prompt(category: str, prompt_type: PromptType, prompt_input: str, prompt_entity: str=None, prompt_entity_start: int=0, shots: ShotType=ShotType.FEW_SHOT):
    if not prompt_type in PromptType:
        raise ValueError(f"Expected PromptType, got '{prompt_type}'")

    def _substitute(base_template: Template, example_template: Template, info: PromptInfo):
        substituted = [base_template.substitute(info)]
        for example in info.example:
            substituted.append(example_template.substitute({**info, **example}))
        substituted.append(example_template.substitute(
            info, 
            input = annotate(prompt_input, [dict(
                start_offset = prompt_entity_start,
                end_offset = prompt_entity_start+len(prompt_entity)
            )]) if prompt_entity is not None else prompt_input,
            entity = prompt_entity,
            output = ''))
        return "\n\n".join(substituted)

    if prompt_type == PromptType.ANNOTATE:
        return _substitute(
            prompt_template[prompt_type]['base'],
            prompt_template[prompt_type]['example'],
            get_info(category, prompt_type, annotate_default, num_examples=(shots*shots + shots) // 2)
        )
    elif prompt_type == PromptType.CLASSIFY:
        if category == 'confidential_status':
            return _substitute(
                prompt_template[prompt_type]['base'],
                prompt_template[prompt_type]['example'],
                get_info(
                    category, 
                    prompt_type, 
                    confidential_status_default, 
                    optionify(confidential_statuses),
                    num_examples=(shots*shots + shots) // 2
                )
            )
        elif category == 'identifier_type':
            return _substitute(
                prompt_template[prompt_type]['base'],
                prompt_template[prompt_type]['example'],
                get_info(
                    category, 
                    prompt_type, 
                    identifier_type_default, 
                    optionify(identifier_types),
                    num_examples=(shots*shots + shots) // 2
                )
            )
        else:
            raise ValueError(f'Unknown category for prompt type {prompt_type}: {category}')
    elif prompt_type == PromptType.VERIFY:
        return _substitute(
            prompt_template[prompt_type]['base'],
            prompt_template[prompt_type]['example'],
            get_info(category, prompt_type, verify_default, num_examples=shots)
        )

print('====== Annotate =======')
print(generate_prompt(
    'PERSON', 
    PromptType.ANNOTATE,
    'He has moved out on his own but still keeps some contact with his dad, mainly because he wants to wait until Maria leaves before cutting ties completely.',
    shots=ShotType.ONE_SHOT))
print('====== Classify =======')
print(generate_prompt(
    'confidential_status', 
    PromptType.CLASSIFY,
    'He has moved out on his own but still keeps some contact with his dad, mainly because he wants to wait until Maria leaves before cutting ties completely.', 
    'Maria', 109,
    ShotType.ONE_SHOT))
print('====== Verify =======')
print(generate_prompt(
    'CODE', 
    PromptType.VERIFY,
    'He has moved out on his own but still keeps some contact with his dad, mainly because he wants to wait until Maria leaves before cutting ties completely.',
    'dad', 66,
    ShotType.FEW_SHOT))
