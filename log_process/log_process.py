from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from configuration.config import data_dir

p_output = (Path(data_dir)/'log_0709_processed.csv').open('w')
p_output.write('session_id,send_time,user_type,content_text\n')

p = (Path(data_dir)/'log_0709.csv').open()
p.readline()

group = defaultdict(list)
for l in tqdm(p):
    l = l.strip().split(',')

    group[l[0]].append(l)

for session, ds in tqdm(group.items()):
    if all(d[2]=='客户' for d in ds):
        continue
    for d in ds:
        p_output.write(','.join(d) + '\n')







# df = pd.read_csv(Path(data_dir)/'log_0709.csv', date_parser='send_time')
#
# for k in df.columns.values:
#     df[k] = df[k].astype()
#
#
# df_group = df.groupby(by='session_id')
#
# l = []
# for idx, arr in tqdm(df_group.indices.items()[100]):
#     if any(df.loc[i]['user_type'] == '客服' for i in arr):
#         l.extend([df.loc[i] for i in arr])
#
# new_df = pd.concat(l, axis=1).transpose()
# new_df.to_csv(Path(data_dir)/'log_0709_processed.csv', index=False)


print('Done')