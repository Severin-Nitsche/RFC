import json
import csv

from . import config

with open(config.ANALYZED_POSTS, 'r') as sentiment_file,\
  open(config.PRIVACY_POSTS, 'r') as privacy_file,\
  open(config.REDDIT_POSTS, 'r') as reddit_file:
  posts = json.load(reddit_file)
  privacy = json.load(privacy_file)
  sentiment = json.load(sentiment_file)

sentiments = []
for sent in sentiment:
  for privacy_score, aspect, sentiment_ranking in zip(sent['analysis']['privacy'], sent['analysis']['aspect'], sent['analysis']['sentiment']):
    if privacy_score > .3:
      sentiments.append([
        sent['id'],
        sent['data']['author'],
        aspect,
        privacy_score,
        sentiment_ranking
      ])

privacies = []
for priv in privacy:
  privacies.append([
    posts[priv['index']]['id'],
    posts[priv['index']]['data']['author'],
    priv['content']
  ])

with open(config.SENTIMENT_CSV, 'w') as s:
  csv.writer(s).writerows(sentiments)

with open(config.PRIVACY_CSV, 'w') as p:
  csv.writer(p).writerows(privacies)
