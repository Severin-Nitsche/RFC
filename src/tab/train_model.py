from .. import config

import torch
from torch import nn
from torch.optim import AdamW
from transformers import LongformerTokenizerFast
from .toy_data_handling import *
from torch.utils.data.dataloader import DataLoader
from .longformer_model import Model
from .data_manipulation import process_echr, process_reddit
import tqdm

# With thanks from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
def train_one_epoch(model, optimizer, loss_fn, train_loader):
    running_loss = 0
    last_loss = 0

    for X in tqdm.tqdm(train_loader):
        labels = X['labels']
        optimizer.zero_grad()
        outputs = model(X).permute(0,2,1)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        last_loss = loss.item()
        running_loss += last_loss
    return last_loss, running_loss / len(train_loader)

def get_loader(dataset, tokenizer, label_set=LabelSet(labels=["MASK"]), include_annotations=True, tokens_per_batch=4096, batch_size=1, hint='echr', label=None):
    if isinstance(dataset, str): # We have to load the dataset
        print(f'Loading dataset @{dataset}')
        process = None
        if hint == 'echr':
            process = process_echr
        elif hint == 'reddit':
            process = process_reddit
        else:
            raise ValueError(f'Unknown hint "{hint}", expected echr or reddit')
        if label is None:
            dataset = process(dataset)
        else:
            dataset = process(dataset, label)
    dataset = WindowedDataset(
        data=dataset, 
        tokenizer=tokenizer,
        label_set=label_set,
        include_annotations=include_annotations,
        tokens_per_batch=tokens_per_batch
    )
    return DataLoader(dataset, collate_fn=WindowBatch, batch_size=batch_size)

if __name__ == '__main__':
    bert = 'allenai/longformer-base-4096'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device}')

    tokenizer = LongformerTokenizerFast.from_pretrained(bert)
    label_set=LabelSet(labels=["MASK"])

    model = Model(model = bert, num_labels = label_set.num_labels())
    model.to(device)

    if device == 'cuda':
        criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.Tensor([1.0, 10.0, 10.0]).cuda())
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.Tensor([1.0, 10.0, 10.0]))
    
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    base_loader = get_loader(config.ECHR, tokenizer)
    fine_loader = get_loader(config.CONVERTED_POSTS, tokenizer)

    losses, epochs = [], []
    for epoch in range(5):
        print('Base-Epoch: ', epoch + 1)
        epochs.append(epoch)
        model.train()
        last_loss, avg_loss = train_one_epoch(model, optimizer, criterion, base_loader)
        print('Training loss (last/avg): {0:.2f}/{1:.2f}'.format(last_loss, avg_loss))
    for epoch in range(2):
        print('Fine-Epoch: ', epoch + 1)
        epochs.append(epoch)
        model.train()
        last_loss, avg_loss = train_one_epoch(model, optimizer, criterion, fine_loader)
        print('Training loss (last/avg): {0:.2f}/{1:.2f}'.format(last_loss, avg_loss))

    torch.save(model.state_dict(), config.MODEL)
