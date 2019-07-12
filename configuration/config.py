import os
from pathlib import Path

if os.environ.get("ROOT_DIR"):
    ROOT_PATH = "/root/insurance_bot"
else:
    ROOT_PATH = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

data_dir = os.path.join(ROOT_PATH, "data")


""" bert """
bert_data_path = Path.home()/'.pytorch_pretrained_bert'
bert_vocab_path = bert_data_path / 'bert-base-chinese-vocab.txt'
bert_model_path = bert_data_path / 'bert-base-chinese.tar.gz'

tencent_w2v_path = Path.home()/'.word2vec'