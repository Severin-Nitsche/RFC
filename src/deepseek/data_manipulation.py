from .. import config

from .deepseek_types import PromptType, Example, PromptInfo

import json
# from nlp import nlp
import tqdm

def annotate(text, entities):
    entities.sort(key=lambda mention: mention['start_offset'])
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
    This function converts the echr files to usable examples with the deepseek templates.
    """
    if not prompt_type in PromptType:
        raise ValueError(f"Expected PromptType, got '{prompt_type}'")

    flatten = []

    def _process(func):
        for data in echr:
            for annotator in data['annotations']:
                func(data, data['text'], annotator, category)
    
    def _format(data, text, annotator, category, _map, _filter, _post):
        entities = list(map(
            _map(data, text, annotator, category),
            filter(_filter(data, text, annotator, category),
            data['annotations'][annotator]['entity_mentions'])))
        return _post(data, text, annotator, category, entities)
    
    _format_wrapper = lambda _map, _filter, _post=(lambda _, __, ___, ____, x: x):\
        lambda data, text, annotator, category:\
            _format(data, text, annotator, category, _map, _filter, _post)

    def _annotate_post(data, text, annotator, category, entities):
        return Example(input=text, output=annotate(text, entities))

    _format_annotate = _format_wrapper(
        lambda data, text, annotator, category: lambda mention:
            {
                'start_offset': mention['start_offset'],
                'end_offset': mention['end_offset']
            },
        lambda data, text, annotator, category: lambda mention: 
            mention['entity_type'] == category,
        _annotate_post
    )

    _format_verify = _format_wrapper(
        lambda data, text, annotator, category: lambda mention:
            Example(
                input=annotate(text, [mention]),
                output='yes' if mention['entity_type'] == category else 'no',
                entity=text[mention['start_offset']:mention['end_offset']]
            ),
        lambda data, text, annotator, category: lambda mention: True,
    )

    _format_classify = _format_wrapper(
        lambda data, text, annotator, category: lambda mention:
            Example(
                input=annotate(text, [mention]),
                output=mention[category],
                entity=text[mention['start_offset']:mention['end_offset']]
            ),
        lambda data, text, annotator, category: lambda mention: True,
    )

    if prompt_type == PromptType.ANNOTATE:
        _process(lambda data, text, annotator, category: flatten.append(_format_annotate(data, text, annotator, category)))
    elif prompt_type == PromptType.CLASSIFY:
        _process(lambda data, text, annotator, category: flatten.extend(_format_classify(data, text, annotator, category)))
    elif prompt_type == PromptType.VERIFY:
        _process(lambda data, text, annotator, category: flatten.extend(_format_verify(data, text, annotator, category)))

    return flatten
