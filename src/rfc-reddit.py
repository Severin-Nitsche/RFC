import praw
import spacy
import json

import config

# Get instance of reddit api
reddit = praw.Reddit(
    client_id=config.CLIENT_ID,
    client_secret=config.CLIENT_SECRET,
    user_agent=config.USER_AGENT
)

posts = {}
# Ggf. python -m spacy download en_core_web_sm
nlp = nlp = spacy.load("en_core_web_sm")

def process(submission):
    if submission.is_self:
        if submission.author.id not in posts:
            posts[submission.author.id] = {
                'posts': []
            }
        doc = nlp(submission.selftext)
        posts[submission.author.id]['posts'].append({
            'title': submission.title,
            'sentences': [sent.text for sent in doc.sents]
        })

for submission in reddit.subreddit("relationship_advice").hot(limit=10):
    if submission.author is None: # Trust me, this case can happen
        continue
    # process(submission) # This leads to duplicates
    for submission_a in submission.author.submissions.hot(limit=10): # TODO: Consider .comments
        process(submission_a)

# Save the dictionary to a JSON file
with open(config.REDDIT_POSTS, "w") as json_file:
    json.dump(posts, json_file)