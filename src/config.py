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

HPC_WORK = os.getenv("HPCWORK")
CACHE_DIR = '/cache'

if HPC_WORK is None:
    warn("$HPCWORK not defined; not changing cache directory (used for huggingface models)")
else:
    os.environ['HF_HOME'] = HPC_WORK + CACHE_DIR
    # Make sure to import this before huggingface
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # Pytorch memory combat