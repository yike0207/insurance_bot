import collections
import json
import logging
import pickle
import random
from pathlib import Path

import gensim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from configuration.config import data_dir

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
train_labels, train_utts, train_tokens = zip(*[(l['label'], l['text'], l['text_words']) for l in json.load(train_data)])
dev_labels, dev_utts, dev_tokens = zip(*[(l['label'], l['text'], l['text_words']) for l in json.load(dev_data)])

if not (Path(data_dir)/'kg_labels.json').exists():
    label_list = sorted(set(train_labels))
    json.dump(label_list, (Path(data_dir)/'kg_labels.json').open('w'), ensure_ascii=False)
else:
    label_list = json.load((Path(data_dir)/'kg_labels.json').open())
num_class = len(label_list)
label2id = {l:i for i, l in enumerate(label_list)}

def seq_padding(X):
    L = list(map(len, X))
    ML = max(L)
    return np.array([x + [0]*(ML-len(x)) for x in X])


wv_model = gensim.models.KeyedVectors.load(str(Path(data_dir) / 'tencent_embed_for_insurance'))
word2vec = wv_model.wv.syn0
word_size = word2vec.shape[1]
word2vec = np.concatenate([np.zeros((1, word_size)), np.zeros((1, word_size)), word2vec])  # [word_size+2,200]
id2word = {i + 2: j for i, j in enumerate(wv_model.wv.index2word)}
word2id = {j: i for i, j in id2word.items()}


def seq2vec(token_ids):
    V = []
    for s in token_ids:
        V.append([])
        for w in s:
            for _ in w:
                V[-1].append(word2id.get(w, 1))
    V = seq_padding(V)
    V = word2vec[V]
    return V


class data_generator:
    def __init__(self, data_x, data_y, data_xw):
        self.data_x = data_x
        self.data_y = data_y
        self.data_xw = data_xw
        self.steps = len(self.data_x) // batch_size
        if len(self.data_x) % batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        idxs = list(range(len(self.data_x)))
        np.random.shuffle(idxs)
        Y,T,XW = [], [],[]
        for i in idxs:
            text = self.data_x[i]
            label = self.data_y[i]
            text_words = self.data_xw[i]

            Y.append(label2id.get(label))
            T.append(text)
            XW.append(text_words)

            if len(Y) == batch_size or i == idxs[-1]:
                Y = torch.tensor(Y, dtype=torch.long)
                XW = torch.tensor(seq2vec(XW), dtype=torch.float32)

                yield Y,XW,T
                Y,XW,T = [],[],[]



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


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()

        self.conv = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=200, kernel_size=(k, 200), padding=(k - 1, 0)) for k in [2,3,4]])

        self.linear = nn.Linear(3 * 200, num_class)  # 3 * 128 -> num_class

        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x_emb):
        xs = [F.relu(conv(x_emb.unsqueeze(1))).squeeze(3) for conv in self.conv]
        xm = [F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2) for x in xs]
        xc = torch.cat(xm, 1)
        o = self.dropout(xc)
        o = self.linear(o)

        return o, xc


class HAN(nn.Module):
    def __init__(self):
        super(HAN, self).__init__()
        self.gru = nn.GRU(input_size=200, hidden_size=200, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(in_features=400, out_features=num_class)

    def forward(self, x_emb):
        output, h_n = self.gru(x_emb)  # [b,s,h*2] [2,b,200]
        h_n = torch.cat([h_n[0], h_n[1]], dim=-1)
        att = torch.bmm(output, h_n.unsqueeze(2)).squeeze(2)
        att = torch.softmax(att, dim=-1)  # [b, s]

        e = torch.bmm(att.unsqueeze(1), output).squeeze(1)

        o = self.linear(e)

        return o, e


# model = TextCNN()
model = HAN()
model.to(device)

if n_gpu > 1:
    model = torch.nn.DataParallel(model)


# optim
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

trainset = data_generator(train_utts, train_labels, train_tokens)
devset = data_generator(dev_utts, dev_labels, dev_tokens)
best_score = 0
best_epoch = 0
err_list = []
sel_col = ['保障范围_保险责任','核赔','核保','核保_不可投保疾病','保障范围_保险条款',
           '查询_保单','核保_不可投保职业','保全','保障范围_保费费率_属性','保障范围_保额_属性']
for e in range(epoch_num):
    model.train()
    tr_loss = 0

    for batch_idx, batch in enumerate(trainset):
        # if batch_idx > 1:
        #     break
        batch = tuple(t.to(device) if i<len(batch)-1 else t for i, t in enumerate(batch))
        y_ids, x_w, _ = batch
        y_p, _ = model(x_w)
        loss = loss_func(y_p, y_ids)

        if n_gpu > 1:
            loss = loss.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad()

        tr_loss += loss.item()
        if batch_idx % 10 == 0 and batch_idx != 0:
            logger.info(f'Epoch:{e} - batch:{batch_idx}/{trainset.steps} - loss: {tr_loss / (batch_idx+1):.8f}')

    model.eval()
    dev_size = 0
    dev_acc = 0
    train_embed_dic = collections.defaultdict(list)
    for batch_idx, batch in enumerate(devset):
        batch = tuple(t.to(device) if i<len(batch)-1 else t for i, t in enumerate(batch))
        y_ids, x_w, ts = batch
        with torch.no_grad():
            y_p, _ = model(x_w)
        for y_id, p in zip(y_ids, y_p):
            y = label_list[y_id]
            if y not in sel_col:
                continue
            train_embed_dic[y].append(p.detach().cpu().numpy())
        y_p = y_p.detach().cpu().numpy()
        y_p = np.argmax(y_p, axis=1)
        y_ids = y_ids.detach().cpu().numpy()

        dev_acc += np.sum(y_p == y_ids)
        dev_size += x_w.size(0)

        for pr, tr, t in zip(y_p, y_ids, ts):
            if pr != tr:
                err_list.append({
                    'epoch_num':e,
                    'text':t,
                    'true':label_list[tr],
                    'pred':label_list[pr]
                })
    pickle.dump(train_embed_dic, Path('embed_han.pkl').open('wb'))

    acc = dev_acc/dev_size
    if acc > best_score:
        best_score = acc
        best_epoch = e

        # save
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), Path('kg_intent_model_han.pt'))

        json.dump(err_list, Path('err_han.json').open('w'), ensure_ascii=False, indent=4)

    logger.info(f'Epoch:{e} - dev acc:{acc:.6f} - best_score:{best_score:.4f} - best_epoch:{best_epoch}')










