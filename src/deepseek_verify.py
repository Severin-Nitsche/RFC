if not __name__ == '__main__':
    exit(0)

# Credit to https://michaelwornow.net/2024/01/09/lm-format-enforcer-demo
import config

from deepseek import deepseek

from deepseek_types import PromptType
from prompts import generate_prompt, entity_types, identifier_types, confidential_statuses

from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams

import json

def _get_verify_inputs():
    inputs = []
    with open(config.TAGGED_POSTS, 'r', encoding='utf-8') as file:
        inputs = json.load(file)
    return inputs

def _get_verify_prompts_and_meta(inputs):
    prompts = [generate_prompt(
        annotation['category'],
        PromptType.VERIFY,
        annotation['input'],
        tag['tag'],
        tag['pos'],
        shots=config.SHOTS
    ) for annotation in inputs for tag in annotation['tags']]
    meta = [dict(
        id = annotation['id'],
        input = annotation['input'],
        tag = tag,
        category = annotation['category']
    ) for annotation in inputs for tag in annotation['tags']]
    return prompts, meta

def _get_verify_sampling_params(prompts, metas, model):
    return SamplingParams(
        guided_decoding = GuidedDecodingParans(choice=['yes', 'no']))

def _serialize_verify_result(result, meta):
    return dict(
        id = meta['id'],
        input = meta['input'],
        category = meta['category'],
        tag = meta['tag'],
        output = result.outputs[0].text
    )

deepseek(
    _get_verify_inputs,
    _get_verify_prompts_and_meta,
    _get_verify_sampling_params,
    config.VERIFIED_POSTS,
    _serialize_verify_result)