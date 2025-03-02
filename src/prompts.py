import config

from data_manipulation import PromptType, get_info, confidential_status_default, identifier_type_default, verify_default, annotate_default, optionify, identifier_types, confidential_statuses

from string import Template

prompt_template = dict()

with open(config.ANNOTATE, 'r') as annotate_file:
    prompt_template[PromptType.ANNOTATE] = Template(annotate_file.read())

with open(config.CLASSIFY, 'r') as classify_file:
    prompt_template[PromptType.CLASSIFY] = Template(classify_file.read())

with open(config.VERIFY, 'r') as verify_file:
    prompt_template[PromptType.VERIFY] = Template(verify_file.read())

def generate_prompt(category: str, prompt_type: PromptType, prompt_input: str, prompt_entity: str=None):
    if not prompt_type in PromptType:
        raise ValueError(f"Expected PromptType, got '{prompt_type}'")

    def _info_to_dict(info):
        res = dict(
            prompt_type = info.prompt_type,
            category = info.category,
            explanation = info.explanation,
            options = info.options,
            prompt_input = prompt_input,
            prompt_entity = prompt_entity
        )
        for i in range(len(info.example)):
            res[f'example_{i}_input'] = info.example[i].input
            res[f'example_{i}_output'] = info.example[i].output
            res[f'example_{i}_entity'] = info.example[i].entity
        return res

    if prompt_type == PromptType.ANNOTATE:
        return prompt_template[prompt_type].substitute(_info_to_dict(
            get_info(category, prompt_type, annotate_default)
        ))
    elif prompt_type == PromptType.CLASSIFY:
        if category == 'confidential_status':
            return prompt_template[prompt_type].substitute(_info_to_dict(get_info(
                category, 
                prompt_type, 
                confidential_status_default, 
                optionify(confidential_statuses)
            )))
        elif category == 'identifier_type':
            return prompt_template[prompt_type].substitute(_info_to_dict(get_info(
                category, 
                prompt_type, 
                identifier_type_default, 
                optionify(identifier_types)
            )))
        else:
            raise ValueError(f'Unknown category for prompt type {prompt_type}: {category}')
    elif prompt_type == PromptType.VERIFY:
        return prompt_template[prompt_type].substitute(_info_to_dict(
            get_info(category, prompt_type, verify_default)
        ))

# print('====== Annotate =======')
# print(generate_prompt('PERSON', PromptType.ANNOTATE,'He has moved out on his own but still keeps some contact with his dad, mainly because he wants to wait until Maria leaves before cutting ties completely.'))
# print('====== Classify =======')
# print(generate_prompt('confidential_status', PromptType.CLASSIFY,'He has moved out on his own but still keeps some contact with his dad, mainly because he wants to wait until Maria leaves before cutting ties completely.', 'Maria'))
# print('====== Verify =======')
# print(generate_prompt('CODE', PromptType.VERIFY,'He has moved out on his own but still keeps some contact with his dad, mainly because he wants to wait until Maria leaves before cutting ties completely.','dad'))