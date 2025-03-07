if not (__name__ == '__main__'):
    exit(0)

from pyabsa import AspectTermExtraction as ATEPC, available_checkpoints
import json
import spacy
import tqdm

from . import config

# you can view all available checkpoints by calling available_checkpoints()
# checkpoint_map = available_checkpoints()

aspect_extractor = ATEPC.AspectExtractor('multilingual')

# Ggf. python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_lg")

privacy = 'privacy security confidentiality cybersecurity'
privacy_tokens = nlp(privacy)

with open(config.REDDIT_POSTS,"r") as posts_file:
    posts = json.load(posts_file)
    for post in tqdm.tqdm(posts):
        doc = nlp(post['data']['text'])
        sentences = [sent.text for sent in doc.sents]
        analysis = aspect_extractor.predict(sentences, save_result=False, print_result=False) # .batch_predict should be fine, too
        post['analysis'] = {
            'aspect': [aspect for sent in analysis for aspect in sent['aspect']],
            'sentiment': [sentiment for sent in analysis for sentiment in sent['sentiment']],
            'confidence': [confidence for sent in analysis for confidence in sent['confidence']],
            'privacy':  [max([aspect_token.similarity(privacy_token) for privacy_token in privacy_tokens for aspect_token in nlp(aspect)]) for sent in analysis for aspect in sent['aspect']]
        };
    with open(config.ANALYZED_POSTS, "w") as json_file:
        json.dump(posts, json_file)