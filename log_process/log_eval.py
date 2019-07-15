import collections
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertConfig, BertForSequenceClassification

from configuration.config import data_dir, bert_vocab_path

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

label_list_path = Path(data_dir)/'kg_labels.json'


label_list = json.load(label_list_path.open())
num_class = len(label_list)

log_data = (Path(data_dir)/'log_0709_processed.csv').open()
log_data.readline()
log_data = [l for l in log_data]

batch_size = 64

greeting = [
    '还有什么可以帮',
    '麻烦稍后对我的服务做一下评价',
    '感谢您对众安保险的支持',
    '以上问题都不是',
    '试试人工服务',
    '{'
]

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
    ML = min(max(L), 150)
    return [x[:ML] + [0]*(ML-len(x)) for x in X]

class data_generator:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        S, U, M, T = [], [],[],[]
        for i, d in enumerate(self.data):
            sess_id, dt, obj_type, text = d.strip().split(',')

            # 过滤掉系统回复
            if any(s in text for s in greeting):
                continue

            if i > 0 and self.data[i-1] == self.data[i]:
                U[-1] += '，' + text
            else:
                U.append(text)
                S.append(sess_id)

            if len(U) > batch_size or i == len(self.data)-1:
                for j, u in enumerate(U):
                    T.append([bert_vocab.get('[CLS]')] + [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in u])
                    M.append([1] * len(T[j]))
                T = torch.tensor(seq_padding(T), dtype=torch.long)
                M = torch.tensor(seq_padding(M), dtype=torch.long)
                Sg = torch.zeros(*T.size(), dtype=torch.long)
                yield S, U, M, Sg, T
                S, U, M, T = [],[],[],[]

eval_data = data_generator(log_data)


kg_model_path = Path(data_dir)/'kg_intent_model.pt'
config_path = Path(data_dir)/'kg_intent_config.json'
config = BertConfig(str(config_path))
model = BertForSequenceClassification(config, num_labels=num_class)
model.load_state_dict(torch.load(kg_model_path, map_location='cpu' if not torch.cuda.is_available() else None))


# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    logger.info(f"let's use {n_gpu} gpu")

model.to(device)

if n_gpu > 1:
    model = nn.DataParallel(model)

maps = json.load((Path(data_dir)/'maps_for_log.json').open())

model.eval()
res_p = Path('log_labels.csv').open('w')
for batch_idx, batch in enumerate(eval_data):
    batch = tuple(t if i<2 else t.to(device) for i, t in enumerate(batch))
    sess, utts, masks, segs, t_ids = batch

    y_p = model(t_ids, segs, masks)
    y_p = torch.argmax(y_p, dim=-1)
    y_p = y_p.detach().cpu().numpy()

    y_p = [maps[label_list[p]] for p in y_p]

    for s, u, p in zip(sess, utts, y_p):
        if p == '负样本':
            continue

        res_p.write(f'{s},{u},{p}\n')






