from .. import config

from vllm import LLM
import random
import json

def deepseek(get_inputs, get_prompts_and_meta, get_sampling_params, save_file, serialize_result):
    inputs = get_inputs()
    print(f'Loaded {len(inputs)} inputs.')

    prompts, meta = get_prompts_and_meta(inputs)
    print(f'Generated {len(prompts)} prompts')
    print(f'A prompt: {random.choice(prompts)}')

    # Here, we convert the prompt into messages
    # This allows us to use the chat template wo/ manually copying it
    messages = [[{
        'role': 'user',
        'content': prompt
    }] for prompt in prompts]

    # TODO: It would speed up waiting time, if we parallelize model loading
    #  and other initialization tasks
    print('Initializing Model...')
    deepseek = LLM(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 
        trust_remote_code=True,  
        tensor_parallel_size=2, 
        distributed_executor_backend='ray', 
        gpu_memory_utilization=0.97,
        # enable_reasoning=think,
        # reasoning_parser="deepseek_r1"
    )

    sampling_params = get_sampling_params(prompts, meta, deepseek)

    print('Performing Inference...')
    results = deepseek.chat(
        messages,
        sampling_params = sampling_params
    )

    print(f'Saving to {save_file}...')
    json_results = [None] * len(results)
    for i in range(len(results)):
        json_results[i] = serialize_result(results[i], meta[i])
    with open(save_file, 'w') as json_file:
        json.dump(json_results, json_file)
