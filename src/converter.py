from . import config

import json

"""
This script converts the deepseek output into echr format
"""

with open(config.CLASSIFIED_POSTS, 'r') as classified_file:
    classified_posts = json.load(classified_file)
with open(config.VERIFIED_POSTS, 'r') as verified_file:
    verified_posts = json.load(verified_file)

aggregated = dict()

for post in classified_posts:
    if post['id'] not in aggregated:
        aggregated[post['id']] = dict(
            annotations = {
                f'deepseek-{config.SHOTS}': dict(
                    entity_mentions = dict()
                )
            },
            text = post['input'],
            doc_id = post['id'],
            dataset_type = f'llm-{config.SHOTS}'
        )
    if f"{post['tag']['tag']}-{post['tag']['pos']}" not in aggregated[post['id']]\
        ['annotations']\
        [f'deepseek-{config.SHOTS}']\
        ['entity_mentions']:
        aggregated[post['id']]\
            ['annotations']\
            [f'deepseek-{config.SHOTS}']\
            ['entity_mentions']\
            [f"{post['tag']['tag']}-{post['tag']['pos']}"] = \
            dict(
                entity_mention_id = f"{post['tag']['tag']}-{post['tag']['pos']}",
                start_offset = post['tag']['pos'],
                end_offset = post['tag']['pos'] + len(post['tag']['tag']),
                span_text = post['tag']['tag'],
                edit_type = 'N/A',
                entity_id = 'N/A'
            )
    aggregated[post['id']]\
        ['annotations']\
        [f'deepseek-{config.SHOTS}']\
        ['entity_mentions']\
        [post['category']] = post['output']

for post in verified_posts:
    if post['output'] == 'yes':
        aggregated[post['id']]\
            ['annotations']\
            [f'deepseek-{config.SHOTS}']\
            ['entity_mentions']\
            ['entity_type'] = post['category']

aggregated[post['id']]\
        ['annotations']\
        [f'deepseek-{config.SHOTS}']\
        ['entity_mentions'] = list(aggregated[post['id']]\
            ['annotations']\
            [f'deepseek-{config.SHOTS}']\
            ['entity_mentions'].values())

aggregated = list(aggregated.values())

with open(config.CONVERTED_POSTS, 'w') as file:
    json.dump(aggregated)