import collections
import json
import random
import re
from pathlib import Path
import numpy as np

from configuration.config import data_dir, bert_vocab_path

p = (Path(data_dir)/'QA对_日志_0712.csv').open()

docs = re.split(r'\n\n', ''.join(p.readlines()))

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

length = []
Q_super_length = []
A_super_length = []
unk_words = []
end_with_Q = []
dic = {}
empty_sess = []


for i, doc in enumerate(docs):
    if not re.search(r'(?<=session_id:).*(?=\n)', doc):
        empty_sess.append(re.search(r'(?<=session_id:).*', doc).group())
        continue
    sess_id = re.search(r'(?<=session_id:).*(?=\n)', doc).group()
    ss = [s[:2] + re.sub(r'[A-Za-z0-9:/.-]','',s[2:].replace(' ||| ', '，')) for s in doc.split('\n') if s!='' and not s.startswith('date') and not s.startswith('sess')]
    for s in ss:
        length.append(len(s))
        unk_words.extend([c for c in s if c not in bert_vocab])
        if len(s) > 250 and s.startswith('A:'):
            A_super_length.append(sess_id)
        if len(s) > 150 and s.startswith('Q:'):
            Q_super_length.append(sess_id)

    if ss[-1].startswith('Q'):
        end_with_Q.append(sess_id)
    dic[sess_id] = ss

unk_words = collections.Counter(unk_words).most_common()
print(f'max: {max(length)}, min: {min(length)}, mean: {np.mean(length)}, median: {np.median(length)}')
print(f'unk words: {len(unk_words)}')
print(unk_words[:100])
print(f'empty sess: {len(empty_sess)}')
print(f'{empty_sess[:10]}')
# json.dump(dic, (Path(data_dir)/'QA_dic.json').open('w'), ensure_ascii=False, indent=4)

print("Done")
