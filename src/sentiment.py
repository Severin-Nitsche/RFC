from pyabsa import AspectTermExtraction as ATEPC, available_checkpoints
import json
import spacy

import config

# you can view all available checkpoints by calling available_checkpoints()
checkpoint_map = available_checkpoints()

aspect_extractor = ATEPC.AspectExtractor('multilingual')

# Ggf. python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

with open(config.REDDIT_POSTS,"r") as posts_file:
    posts = json.load(posts_file)
    for post in posts:
        doc = nlp(post['data']['text'])
        sentences = [sent.text for sent in doc.sents]
        analysis = aspect_extractor.predict(sentences, save_result=False, print_result=False) # .batch_predict should be fine, too
        post['analysis'] = {
            'aspect': [sent['aspect'] for sent in analysis],
            'sentiment': [sent['sentiment'] for sent in analysis],
            'confidence': [sent['confidence'] for sent in analysis],
        };
    with open(config.ANALYZED_POSTS, "w") as json_file:
        json.dump(posts, json_file)