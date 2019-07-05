import collections
import json
import logging
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertConfig

from configuration.config import data_dir, bert_vocab_path
from visualization.model_zoo import BertForSequenceClassification

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

sel_col = ['保障范围_保险责任','核赔','核保','核保_不可投保疾病','保障范围_保险条款',
           '查询_保单','核保_不可投保职业','保全','保障范围_保费费率_属性','保障范围_保额_属性']
train_data = (Path(data_dir) / 'train_data.json').open()
train_data = [(l['label'], l['text']) for l in json.load(train_data) if l['label'] in sel_col]

label_list_path = Path(data_dir)/'kelvin_kg_labels.json'
kg_model_path = Path(data_dir)/'kelvin_kg_intent_model.pt'
config_path = Path(data_dir)/'kelvin_kg_intent_config.json'

label_list = json.load(label_list_path.open())
num_class = len(label_list)

config = BertConfig(str(config_path))
model = BertForSequenceClassification(config, num_labels=num_class)
model.load_state_dict(torch.load(kg_model_path, map_location='cpu' if not torch.cuda.is_available() else None))


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



train_dic = collections.defaultdict(list)
train_embed_dic = collections.defaultdict(list)
for d in train_data:
    label, text = d
    text_ids = [bert_vocab.get('[CLS]')] + [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in text]
    text_mask = [1] * len(text_ids)
    text_ids = torch.tensor([text_ids], dtype=torch.long)
    text_mask = torch.tensor([text_mask], dtype=torch.long)
    text_seg = torch.zeros(*text_ids.size(), dtype=torch.long)
    if len(train_dic[label]) < 1000:
        train_dic[label].append((
            text_ids,
            text_mask,
            text_seg
        ))


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


model.to(device)

if n_gpu > 1:
    model = torch.nn.DataParallel(model)


model.eval()
dev_size = 0
dev_acc = 0
for y, Xs in train_dic.items():
    for x in Xs:
        batch = tuple(t.to(device) for t in x)
        x_ids, x_m, x_s = batch
        with torch.no_grad():
            example_embed = model(x_ids, x_s, x_m)
        train_embed_dic[y].append(example_embed[0].detach().cpu().numpy())

pickle.dump(train_embed_dic, (Path(data_dir)/'embed.pkl').open('wb'))











