from pyabsa import AspectTermExtraction as ATEPC, available_checkpoints
import json
import config

# you can view all available checkpoints by calling available_checkpoints()
checkpoint_map = available_checkpoints()

aspect_extractor = ATEPC.AspectExtractor('multilingual')

with open(config.REDDIT_POSTS,"r") as posts_file:
    posts = json.load(posts_file)
    for author in posts:
        for post in posts[author]['posts']:
            sent_analysis = {
                'aspect': [],
                'sentiment': [],
                'confidence': [],
            };
            analysis = aspect_extractor.predict(post['sentences'], save_result=False, print_result=False) # .batch_predict should be fine, too
            for sent in analysis:
                sent_analysis['aspect'].extend(sent['aspect'])
                sent_analysis['sentiment'].extend(sent['sentiment'])
                sent_analysis['confidence'].extend(sent['confidence'])
            post['analysis'] = sent_analysis
    with open(config.ANALYZED_POSTS, "w") as json_file:
        json.dump(posts, json_file)