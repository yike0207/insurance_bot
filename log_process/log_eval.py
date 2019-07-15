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
log_data_dic = collections.defaultdict(list)
for l in log_data:
    l = l.strip().split(',')
    log_data_dic[l[0]].append(l)

batch_size = 64

greeting = [
    '小马送你的',
    '进入人工系统',
    '有什么可以帮',
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
        S, U, O, M, T = [], [],[],[],[]
        for i, (sess_id, ds) in enumerate(self.data.items()):
            pre_obj_t = ''
            for k, d in enumerate(ds):
                sess_id, dt, obj_type, text = d

                # 过滤掉系统回复
                if any(s in text for s in greeting):
                    continue

                if k > 0 and pre_obj_t == ds[k][2] and len(U)>0:
                    U[-1] += '，' + text
                else:
                    U.append(text)
                    S.append(sess_id)
                    O.append(obj_type)
                    pre_obj_t = obj_type

                if len(U) >= batch_size or i == len(self.data)-1:
                    for j, u in enumerate(U):
                        T.append([bert_vocab.get('[CLS]')] + [bert_vocab.get(c, bert_vocab.get('[UNK]')) for c in u])
                        M.append([1] * len(T[j]))
                    T = torch.tensor(seq_padding(T), dtype=torch.long)
                    M = torch.tensor(seq_padding(M), dtype=torch.long)
                    Sg = torch.zeros(*T.size(), dtype=torch.long)
                    logger.info(f'T:{T.size()}, M:{M.size()}, Sg:{Sg.size()}')
                    yield S, U, O, M, Sg, T
                    S, U, O, M, T = [],[],[],[], []
                    pre_obj_t = ''

eval_data = data_generator(log_data_dic)


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
    batch = tuple(t if i<3 else t.to(device) for i, t in enumerate(batch))
    sess, utts, obj_t, masks, segs, t_ids = batch

    y_p = model(t_ids, segs, masks)
    y_p = torch.argmax(y_p, dim=-1)
    y_p = y_p.detach().cpu().numpy()

    y_p = [maps[label_list[p]] for p in y_p]

    for s, u, o, p in zip(sess, utts, obj_t, y_p):
        if p == '负样本':
            continue

        res_p.write(f'{s},{u},{o},{p}\n')






