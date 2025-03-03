if not __name__ == '__main__':
    exit(0)

# Credit to https://michaelwornow.net/2024/01/09/lm-format-enforcer-demo
import config

from deepseek_types import PromptType
from prompts import generate_prompt, entity_types, identifier_types, confidential_statuses, construct_examples

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParans
import json
import random
import tqdm

from ner_parser import NERParser, parse

inputs = []

with open(config.TAGGED_POSTS, 'r', encoding='utf-8') as file:
    inputs = json.load(file)

print(f'Loaded {len(inputs)} tagged sentences.')

prompts = [generate_prompt(
    prompt.type,
    PromptType.VERIFY,
    prompt.input,
    tag
) for prompt in inputs for tag in prompt['tags']]

print(f'Generated {len(prompts)} VERIFY prompts')

print('Initializing Model...')

deepseek = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", trust_remote_code=True, tensor_parallel_size=2, distributed_executor_backend='mp', gpu_memory_utilization=0.97)

tokenizer_data = build_vllm_token_enforcer_tokenizer_data(deepseek)

tokenizer_data = build_vllm_token_enforcer_tokenizer_data(deepseek)
sampling_params = SamplingParams( # temperature=?, top_p=?, .max_tokens=?
    guided_decoding = GuidedDecodingParams(choice=['yes','no'])
)

print('Performing Verification...')
print(f'Test prompt: {prompts[0]}')
results = deepseek.generate(prompts, sampling_params=sampling_params)
print('Dumping...')
j_res = [None] * len(results)
i = 0
for prompt in inputs for tag in prompt['tags']:
    j_res[i] = dict(
        input = prompt.input,
        output = results[i].outputs[0].text,
        tag = tag
    )
    i += 1

with open(config.VERIFIED_POSTS, "w") as json_file:
    json.dump(j_res, json_file)
# print(results)
