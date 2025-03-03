if not __name__ == '__main__':
    exit(0)

# Credit to https://michaelwornow.net/2024/01/09/lm-format-enforcer-demo
import config

from deepseek_types import PromptType
from prompts import generate_prompt, entity_types, identifier_types, confidential_statuses, construct_examples
from data_manipulation import preprocess

from vllm import LLM, SamplingParams
from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data
import json
import random
import tqdm

from ner_parser import NERParser, parse

echr = preprocess(config.ECHR_DEV, lambda data: data['text'], 'echr')
construct_examples(echr)
posts = preprocess(config.REDDIT_POSTS, lambda post: post['data']['text'], 'Reddit Posts')

inputs = []

for post in posts:
    for sent in post['nlp'].sents:
        inputs.append(sent.text)

print(f'Extracted {len(inputs)} sentences.')

if config.MAX_ANNOTATE_PROMPTS > 0:
    inputs = random.sample(inputs, config.MAX_ANNOTATE_PROMPTS // len(entity_types))

parsers = [NERParser(prompt, config.TAG_START, config.TAG_END) for prompt in inputs for _ in entity_types]
prompts = [generate_prompt(
    category,
    PromptType.ANNOTATE,
    prompt
) for prompt in inputs for category in entity_types]

print(f'Generated {len(prompts)} ANNOTATE prompts')

print('Initializing Model...')

deepseek = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", trust_remote_code=True, tensor_parallel_size=2, distributed_executor_backend='mp', gpu_memory_utilization=0.97)

tokenizer_data = build_vllm_token_enforcer_tokenizer_data(deepseek)

tokenizer_data = build_vllm_token_enforcer_tokenizer_data(deepseek)
sampling_params = [
    SamplingParams( # temperature=?, top_p=?, .max_tokens=?
        logits_processors = [build_vllm_logits_processor(
            tokenizer_data,
            parser
        )]
    ) for parser in parsers
]

print('Performing Inference...')
print(f'Test prompt: {prompts[0]}')
results = deepseek.generate(prompts, sampling_params=sampling_params)
print('============= Results ==============')
j_res = [None] * len(results)
for i in range(len(results)):
    # print(result)
    j_res[i] = dict(
        input = inputs[i],
        output = results[i].outputs[0].text,
        tags = parse(inputs[i], results[i].outputs[0].text)
    )

with open(config.TAGGED_POSTS, "w") as json_file:
    json.dump(j_res, json_file)
# print(results)
