# Credit to https://michaelwornow.net/2024/01/09/lm-format-enforcer-demo
from .. import config

from .deepseek_types import PromptType
from .prompts import generate_prompt, entity_types, identifier_types, confidential_statuses

from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import json

def get_classify_inputs():
    inputs = []
    with open(config.VERIFIED_POSTS, 'r', encoding='utf-8') as file:
        inputs = json.load(file)
    return inputs

def get_classify_prompts_and_meta(inputs):
    annotations = list(filter(
        lambda annotation: annotation['output'] == 'yes',
        inputs
    ))
    prompts = [generate_prompt(
        category,
        PromptType.CLASSIFY,
        annotation['input'],
        annotation['tag']['tag'],
        annotation['tag']['pos'],
        shots=config.SHOTS
    ) for annotation in annotations for category in ['confidential_status', 'identifier_type']]
    meta = [dict(
        id = annotation['id'],
        input = annotation['input'],
        offset = annotation['offset'],
        tag = annotation['tag'],
        category = category
    ) for annotation in annotations for category in ['confidential_status', 'identifier_type']]
    return prompts, meta

def get_classify_sampling_params(prompts, metas, model):
    conf = SamplingParams(
        guided_decoding = GuidedDecodingParams(choice=confidential_statuses))
    ids = SamplingParams(
        guided_decoding = GuidedDecodingParams(choice=identifier_types))
    return [conf if meta['category'] == 'confidential_status' else ids for meta in metas]

def serialize_classify_result(result, meta):
    return dict(
        id = meta['id'],
        input = meta['input'],
        offset = meta['offset'],
        category = meta['category'],
        tag = meta['tag'],
        output = result.outputs[0].text)
