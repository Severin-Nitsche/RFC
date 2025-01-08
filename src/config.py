from dotenv import load_dotenv
from warnings import warn
import platform
import os

VERSION = '1.0.0-alpha'

OUT_DIR = './out'
REDDIT_POSTS = OUT_DIR + '/posts.json'
ANALYZED_POSTS = OUT_DIR + '/processed_posts.json'

load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = f'{platform.system()}:{CLIENT_ID}:{VERSION} (by /u/SeveDaMan)'

if CLIENT_ID is None:
    warn("CLIENT_ID not set in .env (used for reddit API)")
if CLIENT_SECRET is None:
    warn("CLIENT_SECRET not set in .env (used for reddit API)")