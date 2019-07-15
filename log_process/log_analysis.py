from collections import defaultdict, Counter
from pathlib import Path

from configuration.config import data_dir

p = (Path(data_dir)/'log_labels_only_customer.csv').open()
p.readline()
dic = defaultdict(list)
for l in p:
    if len(l.split(',')) > 4:
        continue
    sess, k1,k2,label = l.strip().split(',')
    dic[sess].append(label)

sess_dic = defaultdict(dict)
c1 = defaultdict(int)
c2 = defaultdict(int)
for s, ls in dic.items():
    N = len(ls)
    for l, n in Counter(ls).most_common():
        sess_dic[s][l] = n / N / (ls.index(l) + 1)
    sorted_sess_intent = sorted(sess_dic[s].items(), key=lambda x: x[1], reverse=True)
    c1[sorted_sess_intent[0][0]] += 1
    if len(sorted_sess_intent) > 1:
        co_l = ' ||| '.join(sorted([sorted_sess_intent[0][0], sorted_sess_intent[1][0]]))
        c2[co_l] += 1

for k, v in c1.items():
    print(f'{k},{v}')

print('*'*20)

for k, v in list(c2.items())[:50]:
    print(f'{k},{v}')

print('Done')



