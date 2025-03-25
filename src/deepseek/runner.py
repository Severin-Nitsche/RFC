if not __name__ == '__main__':
    exit(0)

from .. import config

from .deepseek import deepseek

from .deepseek_annotate import get_annotate_inputs, get_annotate_prompts_and_meta, get_annotate_sampling_params, serialize_annotate_result
from .deepseek_verify import get_verify_inputs, get_verify_prompts_and_meta, get_verify_sampling_params, serialize_verify_result
from .deepseek_classify import get_classify_inputs, get_classify_prompts_and_meta, get_classify_sampling_params, serialize_classify_result

# deepseek(
#     get_annotate_inputs,
#     get_annotate_prompts_and_meta,
#     get_annotate_sampling_params,
#     config.TAGGED_POSTS,
#     serialize_annotate_result)

# deepseek(
#     get_verify_inputs,
#     get_verify_prompts_and_meta,
#     get_verify_sampling_params,
#     config.VERIFIED_POSTS,
#     serialize_verify_result)

deepseek(
    get_classify_inputs,
    get_classify_prompts_and_meta,
    get_classify_sampling_params,
    config.CLASSIFIED_POSTS,
    serialize_classify_result)
