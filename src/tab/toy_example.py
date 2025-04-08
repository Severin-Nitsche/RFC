from .. import config

import torch
import json
from transformers import LongformerTokenizerFast
from .toy_data_handling import *
from torch.utils.data.dataloader import DataLoader
from .longformer_model import InferenceModel
# from data_manipulation import dev_raw


bert = "allenai/longformer-base-4096"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = LongformerTokenizerFast.from_pretrained(bert)
label_set = LabelSet(labels=["MASK"])

model = InferenceModel(model = bert, num_labels = label_set.num_labels())
model.to(device)
model.load_state_dict(torch.load(config.MODEL, weights_only = True, map_location = torch.device(device)))
model.eval()

with open(config.REDDIT_POSTS,"r") as posts_file:
    posts_json = json.load(posts_file)
    posts = [post['data'] for post in posts_json]

    toy = WindowedDataset(data=posts, tokenizer=tokenizer, label_set=label_set, include_annotations=False, tokens_per_batch=4096)
    toyloader = DataLoader(toy, collate_fn=WindowBatch, batch_size=1)

    res = []
    o_start = -1
    p_end = -1
    p_ix = -1
    for X in toyloader:
        with torch.no_grad():
            pred = model(X)
            print(pred)
            for ix, offsets, prediction in zip(X.ixs, X.offsets, pred):
                for (start, end), label in zip(offsets, prediction):
                    if label > 0:
                        if (not (ix == p_ix and start - p_end < 5)):
                            if p_ix > -1:
                                res.append({
                                    'index': p_ix,
                                    'content': posts[p_ix]['text'][o_start:p_end]
                                })
                            p_ix = ix
                            o_start = start
                        p_end = end
    res.append({
        'index': p_ix,
        'content': posts[p_ix]['text'][o_start:p_end]
    })
    with open(config.PRIVACY_POSTS, 'w') as file:
        json.dump(res,file)
