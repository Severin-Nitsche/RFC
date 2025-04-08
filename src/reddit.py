import praw
import json

from . import config

import random

# Get instance of reddit api
reddit = praw.Reddit(
    client_id=config.CLIENT_ID,
    client_secret=config.CLIENT_SECRET,
    user_agent=config.USER_AGENT
)

posts = []

def process(submission):
    if submission.is_self:
        posts.append({
            'id': submission.id,
            'data': {
                'author': submission.author.id,
                'title': submission.title,
                'text': submission.selftext
            }
        })

def process_comment(comment):
    posts.append({
        'id': comment.id,
        'data': {
            'author': comment.author.id,
            'title': "Comment N/A",
            'text': comment.body
        }
    })

for submission in reddit.subreddit("relationship_advice").new(limit=5):
    if submission.author is None: # Trust me, this case can happen
        continue
    # process(submission) # This leads to duplicates
    for submission_a in submission.author.submissions.hot(limit=100): # TODO: Consider .comments
        process(submission_a)
    for submission_b in submission.author.comments.hot(limit=100):
        process_comment(submission_b)

# Save the dictionary to a JSON file
with open(config.REDDIT_POSTS, "w") as json_file:
    json.dump(posts, json_file)
