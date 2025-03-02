import config

from deepseek_types import PromptType, Example, PromptInfo

import json
from nlp import nlp
import tqdm

def preprocess(path, accessor, desc):
    with open(path, 'r', encoding='utf-8') as file:
        raw = json.load(file)
        for data in tqdm.tqdm(raw, desc=f'Preprocessing {desc}'):
            data['nlp'] = nlp(accessor(data))
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
    
    def _format(data, sent, annotator, category, _map, _filter, _post):
        start_idx, end_idx = sent.start_char, sent.end_char
        text = sent.text
        entities = list(map(
            _map(data, sent, annotator, category, start_idx, end_idx, text),
            filter(_filter(data, sent, annotator, category, start_idx, end_idx, text),
            data['annotations'][annotator]['entity_mentions'])))
        return _post(data, sent, annotator, category, entities, start_idx, end_idx, text)
    
    _format_wrapper = lambda _map, _filter, _post=(lambda _, __, ___, ____, x, _____, ______, _______: x):\
        lambda data, sent, annotator, category:\
            _format(data, sent, annotator, category, _map, _filter, _post)

    def _annotate_post(data, sent, annotator, category, entities, start_idx, end_idx, text):
        entities.sort(key=lambda mention: mention['start_offset'])
        text = sent.text
        result = ''
        r_idx = 0
        for entity in entities:
            result += text[r_idx:entity['start_offset']]
            result +=  config.TAG_START
            result += text[entity['start_offset']:entity['end_offset']]
            result += config.TAG_END
            r_idx = entity['end_offset']
        result += text[r_idx:]
        return Example(input=text, output=result)

    _format_annotate = _format_wrapper(
        lambda data, sent, annotator, category, start_idx, end_idx, text: lambda mention:
            {
                'start_offset': mention['start_offset'] - start_idx,
                'end_offset': mention['end_offset'] - start_idx
            },
        lambda data, sent, annotator, category, start_idx, end_idx, text: lambda mention: 
            mention['entity_type'] == category and
            mention['start_offset'] >= start_idx and
            mention['end_offset'] <= end_idx,
        _annotate_post
    )

    _format_verify = _format_wrapper(
        lambda data, sent, annotator, category, start_idx, end_idx, text: lambda mention:
            Example(
                input=text,
                output='yes' if mention['entity_type'] == category else 'no',
                entity=text[mention['start_offset'] - start_idx:mention['end_offset'] - start_idx]
            ),
        lambda data, sent, annotator, category, start_idx, end_idx, text: lambda mention: 
            mention['start_offset'] >= start_idx and
            mention['end_offset'] <= end_idx,
    )

    _format_classify = _format_wrapper(
        lambda data, sent, annotator, category, start_idx, end_idx, text: lambda mention:
            Example(
                input=text,
                output=mention[category],
                entity=text[mention['start_offset'] - start_idx:mention['end_offset'] - start_idx]
            ),
        lambda data, sent, annotator, category, start_idx, end_idx, text: lambda mention: 
            mention['start_offset'] >= start_idx and
            mention['end_offset'] <= end_idx,
    )

    if prompt_type == PromptType.ANNOTATE:
        _process(lambda data, sent, annotator, category: flatten.append(_format_annotate(data, sent, annotator, category)))
    elif prompt_type == PromptType.CLASSIFY:
        _process(lambda data, sent, annotator, category: flatten.extend(_format_classify(data, sent, annotator, category)))
    elif prompt_type == PromptType.VERIFY:
        _process(lambda data, sent, annotator, category: flatten.extend(_format_verify(data, sent, annotator, category)))

    return flatten