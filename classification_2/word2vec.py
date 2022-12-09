import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from tqdm import tqdm

from configuration.config import data_dir, tencent_w2v_path

# get tencent embedding from .gz
tmpdir = tempfile.mkdtemp()
with tarfile.open(Path(tencent_w2v_path)/'Tencent_AILab_ChineseEmbedding.tar.gz', 'r:gz') as archive:
    
    import os
    
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(archive, tmpdir)

serialization_dir = tmpdir
filename = [fn for fn in Path(serialization_dir).iterdir() if 'Tencent_AILab_ChineseEmbedding' in fn.name][0]
raw_tencent_embeds = filename.open(errors='ignore')
upt_tencent_embeds = (Path(data_dir)/'Tencent_AILab_ChineseEmbedding_tmp.txt').open('w')
if tmpdir:
    shutil.rmtree(tmpdir)
first_line = raw_tencent_embeds.readline()
print(first_line)


# filter tencent embedding to glove format
word_set = []
for fn in ['train_data.json', 'dev_data.json']:
    word_set.extend([w for l in json.load((Path(data_dir)/fn).open()) for w in l['text_words']])
word_set = set(word_set)
word_cnt = 0
for line in tqdm(raw_tencent_embeds):
    l = line.strip().split()
    if len(l) != 201 or l[0] not in word_set:
        continue
    word_cnt += 1
    upt_tencent_embeds.write(line)

print(word_cnt)  #10068


# export to wv
glove_file = Path(data_dir)/'Tencent_AILab_ChineseEmbedding_tmp.txt'
glove_file = datapath(glove_file)

w2v_file = get_tmpfile(Path(data_dir)/'tmpfile')
glove2word2vec(glove_file, w2v_file)

m = gensim.models.KeyedVectors.load_word2vec_format(w2v_file)
m.save(str(Path(data_dir)/'tencent_embed_for_insurance'))