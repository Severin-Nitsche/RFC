# Credit to https://michaelwornow.net/2024/01/09/lm-format-enforcer-demo
from .. import config

from .deepseek_types import PromptType
from .prompts import generate_prompt, entity_types, identifier_types, confidential_statuses

from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from .data_manipulation import preprocess
from .parser import parse
from .grammar import construct_grammar
import json
import random

def get_annotate_inputs():
    return preprocess(config.REDDIT_POSTS, lambda post: post['data']['text'], 'posts')

def test_prompts(inputs):
    post = random.choice(inputs)
    sent = random.choice(list(post['nlp'].doc.sents))
    prompts = [generate_prompt(
        category,
        PromptType.ANNOTATE,
        sent.text,
        shots=config.SHOTS
    ) for category in entity_types]
    meta = [dict(
        id = post['id'],
        offset = sent.start_char,
        input = sent.text,
        category = category
    ) for category in entity_types]
            
    return prompts, meta


def get_annotate_prompts_and_meta(inputs):
    prompts = [generate_prompt(
        category,
        PromptType.ANNOTATE,
        sent.text,
        shots=config.SHOTS
    ) for post in inputs 
        for category in entity_types 
            for sent in post['nlp'].doc.sents]
    meta = [dict(
        id = post['id'],
        offset = sent.start_char,
        input = sent.text,
        category = category
    ) for post in inputs 
        for category in entity_types 
            for sent in post['nlp'].doc.sents]
    return prompts, meta

def get_annotate_sampling_params(prompts, metas, model):
    return [SamplingParams(
        max_tokens = 2048,
        guided_decoding=GuidedDecodingParams(
            grammar = construct_grammar(meta['input'], config.TAG_START, config.TAG_END)
        )
    ) for meta in metas]

def serialize_annotate_result(result, meta):
    return dict(
        id = meta['id'],
        input = meta['input'],
        offset = meta['offset'],
        category = meta['category'],
        output = result.outputs[0].text,
        tags = parse(result.outputs[0].text, meta['input'])
    )
