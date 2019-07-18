import collections
import json
import re
from pathlib import Path

from configuration.config import data_dir

p = (Path(data_dir)/'QA对_日志_0712.csv').open()

docs = re.split(r'\n\n', ''.join(p.readlines()))

dic = collections.defaultdict(list)

for i, doc in enumerate(docs):
    if not re.search(r'(?<=session_id:).*(?=\n)', doc):
        continue
    sess_id = re.search(r'(?<=session_id:).*(?=\n)', doc).group()
    ss = [s for s in doc.split('\n') if s!='' and not s.startswith('date') and not s.startswith('sess')]
    for k in range(0,len(ss),2):
        if k + 1 == len(ss):
            continue
        if not ss[k].startswith('Q:') or not ss[k+1].startswith('A:'):
            continue

        q, a = ss[k].replace('Q:',''), ss[k+1].replace('A:','')

        q = re.sub(r'[A-Za-z0-9:/.-]{5,}', '', q)
        a = re.sub(r'[A-Za-z0-9:/.-]{5,}', '', a)

        if len(q.replace(' ||| ', '，')) > 150:
            qc = q.split(' ||| ')
            qc = list(set(qc))
            qc_n = qc.copy()
            for idx, e in enumerate(qc):
                if '也许您想问' in e:
                    qc_n.remove(e)
                    continue
                if len(e) == 1:
                    qc_n.remove(e)
                    continue
                if 150 < len(e) <= 300:
                    qc_n.remove(e)
                    qc_n.extend([e[sp:sp + len(e) // 2] for sp in range(0, len(e), len(e) // 2)])
                    continue
                if len(e) > 300:
                    qc_n.remove(e)
                    qc_n.append(e[:100])
        else:
            qc_n = [q.replace(' ||| ', '，')]

        if len(a.replace(' ||| ', '，')) > 250:
            ac = a.split(' ||| ')
            ac = list(set(ac))
            ac_n = ac.copy()
            for idx, e in enumerate(ac):
                if len(e) == 1:
                    ac_n.remove(e)
                if 250 < len(e) <= 500:
                    ac_n.remove(e)
                    ac_n.extend([e[sp:sp+len(e)//2] for sp in range(0, len(e), len(e)//2) if len(e[sp:sp+len(e)//2])>1])
                    continue
                if 500 < len(e) <= 750:
                    ac_n.remove(e)
                    ac_n.extend([e[sp:sp+len(e)//3] for sp in range(0, len(e), len(e)//3) if len(e[sp:sp+len(e)//3])>1])
                    continue
                if 750 < len(e) <= 1000:
                    ac_n.remove(e)
                    ac_n.extend([e[sp:sp+len(e)//4] for sp in range(0, len(e), len(e)//4) if len(e[sp:sp+len(e)//4])>1])
                    continue
                if len(e) > 1000:
                    ac_n.remove(e)
                    ac_n.append(e[:250])

        else:
            ac_n = [a.replace(' ||| ', '，')]

        for i_ in qc_n:
            for j_ in ac_n:
                if len(i_)>150 or len(j_)>250:
                    print('tst')
                assert len(i_)<=150 and len(j_)<=250
                dic[sess_id].append((i_,j_))

print(f'{sum([len(p) for s, p in dic.items()])}')
json.dump(dic, (Path(data_dir)/'QA_pair.json').open('w'), ensure_ascii=False)










print("Done")
