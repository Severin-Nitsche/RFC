if not __name__ == '__main__':
    exit(0)

# Credit to https://michaelwornow.net/2024/01/09/lm-format-enforcer-demo
import config

from vllm import LLM, SamplingParams
from lmformatenforcer import CharacterLevelParser
from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data
import json
import spacy
# import torch

from ner_parser import NERParser

nlp = spacy.load('en_core_web_sm')

# with torch.amp.autocast(enabled=torch.amp, dtype=torch.bfloat16) and torch.no_grad() and 
with open(config.REDDIT_POSTS, "r") as posts_file:
    posts = json.load(posts_file)
    prompts = [list(nlp(posts[0]['data']['text']).sents)[0].text]; # TODO: load from json

    print('Initializing Model...')

    deepseek = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", trust_remote_code=True, tensor_parallel_size=2, distributed_executor_backend='mp', gpu_memory_utilization=0.97)

    tokenizer_data = build_vllm_token_enforcer_tokenizer_data(deepseek)
    sampling_params = SamplingParams() # temperature=?, top_p=?, .max_tokens=?


    parsers = [NERParser(prompt, '@@', '##') for prompt in prompts]
    prompts = [
        f'''
        I am an excellent linguist. The task is to label person entities in the given text. Below are some examples.

        Input: I went ahead and decided to contact my son's mother.
        Output: @@I## went ahead and decided to contact my @@son##'s @@mother##.

        Input: My favorite band is on tour and scheduled a show for Valentine's Day in a city near us.
        Output: @@My## favorite band is on tour and scheduled a show for Valentine's Day in a city near us.

        Input: I wasn't there because I was visiting friends in the Netherlands, but my mom and her sister spent time together, and it brought them closer.
        Output: @@I## wasn't there because I was visiting @@friends## in the Netherlands, but my @@mom## and her @@sister## spent time together, and it brought them closer.

        Input: {prompt}
        Output: 
        ''' for prompt in prompts
    ]

    tokenizer_data = build_vllm_token_enforcer_tokenizer_data(deepseek)
    sampling_params = [
        SamplingParams(
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
    for result in results:
        # print(result)
        print(result.outputs[0].text)
    # print(results)
