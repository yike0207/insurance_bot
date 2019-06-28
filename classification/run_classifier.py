import collections
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertForSequenceClassification, BertAdam

from configuration.config import data_dir, bert_vocab_path, bert_model_path, bert_data_path

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark=True

batch_size = 64
epoch_num = 10
learning_rate = 5e-5
warmup_proportion = 0.1

train_data = (Path(data_dir) / 'train_data.json').open()
dev_data = (Path(data_dir) / 'dev_data.json').open()
train_labels, train_utts = zip(*[(l['label'], l['text']) for l in json.load(train_data)])
dev_labels, dev_utts = zip(*[(l['label'], l['text']) for l in json.load(dev_data)])

if not (Path(data_dir)/'kg_labels.json').exists():
    label_list = sorted(set(train_labels))
    json.dump(label_list, (Path(data_dir)/'kg_labels.json').open('w'), ensure_ascii=False)
else:
    label_list = json.load((Path(data_dir)/'kg_labels.json').open())
num_class = len(label_list)
label2id = {l:i for i, l in enumerate(label_list)}

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


bert_vocab = load_vocab(bert_vocab_path)

def seq_padding(X):
    L = list(map(len, X))
    ML = max(L)
    return [x + [0]*(ML-len(x)) for x in X]



class data_generator:
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.steps = len(self.data_x) // batch_size
        if len(self.data_x) % batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        idxs = list(range(len(self.data_x)))
        np.random.shuffle(idxs)
        X, Y,M,T = [], [],[],[]
        for i in idxs:
            text = self.data_x[i]
            label = self.data_y[i]

            text_ids = [bert_vocab.get('[CLS]')] + [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in text]
            text_mask = [1] * len(text_ids)
            X.append(text_ids)
            M.append(text_mask)
            Y.append(label2id.get(label))
            T.append(text)

            if len(X) == batch_size or i == idxs[-1]:
                X = torch.tensor(seq_padding(X), dtype=torch.long)
                Y = torch.tensor(Y, dtype=torch.long)
                M = torch.tensor(seq_padding(M), dtype=torch.long)
                S = torch.zeros(*X.size(), dtype=torch.long)

                yield X, Y, M, S,T
                X, Y, M,T = [],[],[],[]



# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    logger.info(f"let's use {n_gpu} gpu")

# random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if n_gpu > 1:
    torch.cuda.manual_seed_all(42)


model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=bert_model_path,
                                                          cache_dir=bert_data_path,
                                                          num_labels=num_class)
model.to(device)

if n_gpu > 1:
    model = torch.nn.DataParallel(model)


# optim
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

num_train_optimization_steps = len(train_utts) / batch_size * epoch_num
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=learning_rate,
                     warmup=warmup_proportion,
                     t_total=num_train_optimization_steps)

trainset = data_generator(train_utts, train_labels)
devset = data_generator(dev_utts, dev_labels)
best_score = 0
best_epoch = 0
err_list = []

for e in range(epoch_num):
    model.train()
    tr_loss = 0

    for batch_idx, batch in enumerate(trainset):
        batch = tuple(t.to(device) if i<len(batch)-1 else t for i, t in enumerate(batch))
        x_ids, y_ids, x_m, x_s, _ = batch
        loss = model(x_ids, x_s, x_m, y_ids)

        if n_gpu > 1:
            loss = loss.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        tr_loss += loss.item()
        if batch_idx % 10 == 0 and batch_idx != 0:
            logger.info(f'Epoch:{e} - batch:{batch_idx}/{trainset.steps} - loss: {tr_loss / (batch_idx+1):.8f}')

    model.eval()
    dev_size = 0
    dev_acc = 0
    for batch_idx, batch in enumerate(devset):
        batch = tuple(t.to(device) if i<len(batch)-1 else t for i, t in enumerate(batch))
        x_ids, y_ids, x_m, x_s,tt = batch
        with torch.no_grad():
            y_p = model(x_ids, x_s, x_m)
        y_p = y_p.detach().cpu().numpy()
        y_p = np.argmax(y_p, axis=1)
        y_ids = y_ids.detach().cpu().numpy()

        dev_acc += np.sum(y_p == y_ids)
        dev_size += x_ids.size(0)

        for pr, tr,t_ in zip(y_p, y_ids,tt):
            if pr != tr:
                err_list.append({
                    'epoch_num':e,
                    'text':t_,
                    'true':label_list[tr],
                    'pred':label_list[pr]
                })

    acc = dev_acc/dev_size
    if acc > best_score:
        best_score = acc
        best_epoch = e

        # save
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), Path(data_dir)/'kg_intent_model.pt')
        (Path(data_dir) / 'kg_intent_model.json').open('w').write(model_to_save.config.to_json_string())

        json.dump(err_list, (Path(data_dir)/'err.json').open('w'), ensure_ascii=False, indent=4)

    logger.info(f'Epoch:{e} - dev acc:{acc:.6f} - best_score:{best_score:.4f} - best_epoch:{best_epoch}')










