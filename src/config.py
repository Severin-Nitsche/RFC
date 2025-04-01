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
EXAMPLE_DIR = './examples'
TEMPLATE_DIR = './templates'

TAG_START = '@@'
TAG_END = '##'

ECHR = EXAMPLE_DIR + '/echr_train.json'
TAGGED_POSTS = OUT_DIR + '/tagged_posts.json'
VERIFIED_POSTS = OUT_DIR + '/verified_posts.json'
CLASSIFIED_POSTS = OUT_DIR + '/classified_posts.json'
CONVERTED_POSTS = OUT_DIR + '/converted_posts.json'

ANNOTATE_EXAMPLE = TEMPLATE_DIR + '/ANNOTATE_EXAMPLE.tmpl'
ANNOTATE = TEMPLATE_DIR + '/ANNOTATE.tmpl'
ANNOTATE_TASK = TEMPLATE_DIR + '/ANNOTATE_TASK.tmpl'
CLASSIFY_EXAMPLE = TEMPLATE_DIR + '/CLASSIFY_EXAMPLE.tmpl'
CLASSIFY = TEMPLATE_DIR + '/CLASSIFY.tmpl'
CLASSIFY_TASK = TEMPLATE_DIR + '/CLASSIFY_TASK.tmpl'
VERIFY_EXAMPLE = TEMPLATE_DIR + '/VERIFY_EXAMPLE.tmpl'
VERIFY = TEMPLATE_DIR + '/VERIFY.tmpl'
VERIFY_TASK = TEMPLATE_DIR + '/VERIFY_TASK.tmpl'

SHOTS = 1 # 0 - Zero Shot, 1 - One Shot, 2 - Few Shot

HPC_WORK = os.getenv("HPCWORK")
CACHE_DIR = '/cache'

if HPC_WORK is None:
    warn("$HPCWORK not defined; not changing cache directory (used for huggingface models)")
else:
    os.environ['HF_HOME'] = HPC_WORK + CACHE_DIR
    # Make sure to import this before huggingface (otherwise HF_HOME will have no effect)
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# TAB
MODEL = OUT_DIR + '/long_model.pt'
