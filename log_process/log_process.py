from pathlib import Path

import pandas as pd

from configuration.config import data_dir

df = pd.read_csv(Path(data_dir)/'log_0709.csv')

df_group = df.groupby(by='session_id')



print('Done')