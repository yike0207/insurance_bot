import os
from pathlib import Path

ROOT_PATH = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))


data_dir = os.path.join(ROOT_PATH, "data")


""" bert """
bert_data_path = Path.home()/'.pytorch_pretrained_bert'
bert_vocab_path = bert_data_path / 'bert-base-chinese-vocab.txt'
bert_model_path = bert_data_path / 'bert-base-chinese.tar.gz'
