from dotenv import load_dotenv
from warnings import warn
import platform
import os

# General
OUT_DIR = './out'
load_dotenv()

# Reddit Scraper
VERSION = '1.0.0-alpha'
REDDIT_POSTS = OUT_DIR + '/posts.json'

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USER_AGENT = f'{platform.system()}:{CLIENT_ID}:{VERSION} (by /u/SeveDaMan)'

if CLIENT_ID is None:
    warn("CLIENT_ID not set in .env (used for reddit API)")
if CLIENT_SECRET is None:
    warn("CLIENT_SECRET not set in .env (used for reddit API)")

# Sentiment Analysis
ANALYZED_POSTS = OUT_DIR + '/processed_posts.json'

# Deepseek
MAX_ANNOTATE_PROMPTS = 1000 # 0 means all

EXAMPLE_DIR = './examples'
TEMPLATE_DIR = './templates'

TAG_START = '@@'
TAG_END = '##'

ECHR_DEV = EXAMPLE_DIR + '/echr_dev.json'
TAGGED_POSTS = OUT_DIR + '/tagged_posts.json'

ANNOTATE = TEMPLATE_DIR + '/ANNOTATE.tmpl'
CLASSIFY = TEMPLATE_DIR + '/CLASSIFY.tmpl'
VERIFY = TEMPLATE_DIR + '/VERIFY.tmpl'

HPC_WORK = os.getenv("HPCWORK")
CACHE_DIR = '/cache'

if HPC_WORK is None:
    warn("$HPCWORK not defined; not changing cache directory (used for huggingface models)")
else:
    os.environ['HF_HOME'] = HPC_WORK + CACHE_DIR
    # Make sure to import this before huggingface (otherwise HF_HOME will have no effect)
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'