from .. import config

from vllm import LLM
import random
import json

def deepseek(get_inputs, get_prompts_and_meta, get_sampling_params, save_file, serialize_result):
    inputs = get_inputs()
    print(f'Loaded {len(inputs)} inputs.')

    prompts, meta = get_prompts_and_meta(inputs)
    print(f'Generated {len(prompts)} prompts')

    print('Initializing Model...')
    deepseek = LLM(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        trust_remote_code=True,
        tensor_parallel_size=2,
        distributed_executor_backend='mp',
        gpu_memory_utilization=0.97)

    sampling_params = get_sampling_params(prompts, meta, deepseek)

    print('Performing Inference...')
    print(f'A prompt: {random.choice(prompts)}')
    results = deepseek.generate(
        prompts,
        sampling_params = sampling_params)

    print(f'Saving to {save_file}...')
    json_results = [None] * len(results)
    for i in range(len(results)):
        json_results[i] = serialize_result(results[i], meta[i])
    with open(save_file, 'w') as json_file:
        json.dump(json_results, json_file)