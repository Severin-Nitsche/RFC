if not __name__ == '__main__':
    exit(0)

# Credit to https://michaelwornow.net/2024/01/09/lm-format-enforcer-demo
import config

from deepseek import deepseek

from data_manipulation import preprocess

from deepseek_types import PromptType
from prompts import generate_prompt, entity_types, identifier_types, confidential_statuses, construct_examples

from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import json

def _get_classify_inputs():
    echr = preprocess(config.ECHR_DEV, lambda data: data['text'], 'echr')
    construct_examples(echr)
    inputs = []
    with open(config.VERIFIED_POSTS, 'r', encoding='utf-8') as file:
        inputs = json.load(file)
    return inputs

def _get_classify_prompts_and_meta(inputs):
    transformed = dict()
    for annotation in inputs:
        if annotation['id'] not in transformed:
            transformed[annotation['id']] = dict()
        if annotation['input'] not in transformed[annotation['id']]:
            transformed[annotation['id']][annotation['input']] = dict(
                offset = annotation['offset'],
                tags = [])
        if annotation['output'] == 'yes':
            transformed[annotation['id']][annotation['input']]['tags'].append(annotation['tag'])
    prompts = [generate_prompt(
        category,
        PromptType.CLASSIFY,
        sent,
        tag['tag']
    ) for id in transformed for sent in transformed[id] for tag in transformed[id][sent]['tags'] for category in ['confidential_status', 'identifier_type']]
    meta = [dict(
        id = id,
        input = sent,
        tag = tag,
        category = category,
        offset = transformed[id][sent]['offset']
    ) for id in transformed for sent in transformed[id] for tag in transformed[id][sent]['tags'] for category in ['confidential_status', 'identifier_type']]
    return prompts, meta

def _get_classify_sampling_params(prompts, metas, model):
    conf = SamplingParams(
        guided_decoding = GuidedDecodingParams(choice=confidential_statuses))
    ids = SamplingParams(
        guided_decoding = GuidedDecodingParams(choice=identifier_types))
    return [conf if meta['category'] == 'confidential_status' else ids for meta in metas]

def _serialize_classify_result(result, meta):
    return dict(
        id = meta['id'],
        input = meta['input'],
        category = meta['category'],
        tag = meta['tag'],
        offset = meta['offset'],
        output = result.outputs[0].text)

deepseek(
    _get_classify_inputs,
    _get_classify_prompts_and_meta,
    _get_classify_sampling_params,
    config.CLASSIFIED_POSTS,
    _serialize_classify_result)