from pathlib import Path

import pandas as pd

from configuration.config import data_dir


def json2csv():
    df = pd.read_json(Path('train_data.json').open())

    col = ['text','label']
    df[col].to_csv(Path('train_data.csv').open('w'), index=False)

    df = pd.read_json(Path('dev_data.json').open())

    col = ['text','label']
    df[col].to_csv(Path('dev_data.csv').open('w'), index=False)


def json2csv_():
    df = pd.read_json((Path(data_dir)/'err_v4_0712.json').open())

    col = ['epoch_num', 'text', 'true', 'pred']
    df[col][df['epoch_num']==9].to_csv((Path(data_dir)/'err.csv').open('w'), index=False)


if __name__ == '__main__':
    json2csv_()