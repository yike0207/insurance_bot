from pathlib import Path

import pandas as pd

from configuration.config import data_dir

df = pd.read_csv(Path(data_dir)/'log_0709.csv')

df_group = df.groupby(by='session_id')

l = []
for idx, arr in  df_group.indices.items():
    if any(df.loc[i]['user_type'] == '客服' for i in arr):
        l.append([df.loc[i] for i in arr])




print('Done')