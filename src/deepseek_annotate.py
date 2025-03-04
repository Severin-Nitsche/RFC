if not __name__ == '__main__':
    exit(0)

# Credit to https://michaelwornow.net/2024/01/09/lm-format-enforcer-demo
import config

from deepseek import deepseek

from deepseek_types import PromptType
from prompts import generate_prompt, entity_types, identifier_types, confidential_statuses, construct_examples
from data_manipulation import preprocess

from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data

from ner_parser import NERParser, parse

def _get_annotate_inputs():
    echr = preprocess(config.ECHR_DEV, lambda data: data['text'], 'echr')
    construct_examples(echr)
    return preprocess(config.REDDIT_POSTS, lambda post: post['data']['text'], 'Reddit Posts')

def _get_annotate_prompts_and_meta(inputs):
    prompts = [generate_prompt(
        category,
        PromptType.ANNOTATE,
        sent.text
    ) for post in inputs for sent in post['nlp'].sents for category in entity_types]
    meta = [dict(
        id = post['id'],
        input = sent,
        category = category
    ) for post in inputs for sent in post['nlp'].sents for category in entity_types]
    return prompts, meta

def _get_annotate_sampling_params(prompts, metas, model):
    tokenizer_data = build_vllm_token_enforcer_tokenizer_data(model)
    sampling_params = [SamplingParams(
        logits_processors = [build_vllm_logits_processor(
            tokenizer_data,
            NERParser(
                meta['input'],
                config.TAG_START,
                config.TAG_END)
        )]
    ) for meta in metas]

def _serialize_annotate_result(result, meta):
    return dict(
        id = meta['id'],
        input = meta['input'],
        category = meta['category'],
        tags = parse(meta['input'], result.outputs[0].text)
    )

deepseek(
    _get_annotate_inputs,
    _get_annotate_prompts_and_meta,
    _get_annotate_sampling_params,
    config.TAGGED_POSTS)