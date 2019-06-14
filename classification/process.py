import json
from collections import Counter
from pathlib import Path
import numpy as np

import jieba
import pandas as pd

from configuration.config import data_dir


df = pd.read_csv((Path(data_dir)/'raw_0614.csv').open())

label_list = sorted(list(set(df['label'].values)))
json.dump(label_list+['负样本'], (Path(data_dir)/'kg_labels.json').open('w'), ensure_ascii=False)

label_dict = {l: n for l, n in Counter(df['label'].values).most_common()}

train_data, dev_data = [],[]

for row in df.iterrows():
    text = row[1]['question']
    text_words = jieba.lcut(text, HMM=False)
    label = row[1]['label']

    if len(text) > 500:  # 检查句长
        continue

    d = {
        'text': text,
        'text_words': text_words,
        'label': label
    }

    r = np.random.randint(label_dict[label])
    if r < label_dict[label] * 0.8:
        train_data.append(d)
    else:
        dev_data.append(d)

print(f'train size: {len(train_data)}, dev size: {len(dev_data)}')  # train size: 28364, dev size: 7148

# negative samples
neg_list = []
for fn in Path(data_dir).iterdir():
    if str(fn).endswith('csv'):
        continue
    for line in fn.open():
        neg_list.append(line.strip())

idxs = list(range(len(neg_list)))
np.random.shuffle(idxs)
train_neg_list = []
for l in np.array(neg_list)[idxs][:3000]:
    train_neg_list.append({'text':l, 'text_words':jieba.lcut(l, HMM=False),'label':'负样本'})

# save
json.dump(train_data+train_neg_list, (Path(data_dir)/'train_data.json').open('w'), indent=4, ensure_ascii=False)
json.dump(dev_data, (Path(data_dir)/'dev_data.json').open('w'), indent=4, ensure_ascii=False)


