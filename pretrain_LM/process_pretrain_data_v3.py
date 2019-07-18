import collections
import json
import logging
import os
import random
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path
from difflib import SequenceMatcher

from tqdm import tqdm

from configuration.config import data_dir, bert_vocab_path

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

masked_lm_prob = 0.15

max_predictions_per_seq = 20

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", 'label'])


def create_masked_lm_predictions(T):
    cand_indices = []
    for (i, token) in enumerate(T):
        if token in ['[CLS]', '[SEP]']:
            continue
        cand_indices.append(i)

    random.shuffle(cand_indices)
    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(T) * masked_lm_prob))))
    masked_lms, covered_indexes = [], set()
    for index in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        if len(masked_lms) + 1 > num_to_mask:
            continue
        if index in covered_indexes:
            continue
        covered_indexes.add(index)
        masked_token = None

        # 80% of the time , replace with [MASK]
        if random.random() < 0.8:
            masked_token = '[MASK]'

        else:
            # 10% of the time, keep original
            if random.random() < 0.5:
                masked_token = T[index]
            # 10% of the time, replace with random word
            else:
                masked_token = random.choice(list(bert_vocab.keys()))

        masked_lms.append(MaskedLmInstance(index=index, label=T[index]))
        T[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return T, mask_indices, masked_token_labels


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


qa_pair = json.load((Path(data_dir) / 'QA_pair.json').open())

sess_pool = list(qa_pair.keys())

data = list(qa_pair.items())

def next_sentence(d):
    tmp_instance = []
    sess, pairs = d
    sel_sess_pool = random.sample(sess_pool, k=20)
    sel_answer_pool = [p[1] for sid in sel_sess_pool if sid!=sess for p in qa_pair[sid]]
    for pair in pairs:
        q, a = pair
        tokens_a = [c for c in q]

        tokens_b = [c for c in a]

        tokens = ["[CLS]"] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]
        tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(tokens)
        instance = {
            'tokens': tokens,
            'segment_ids': segment_ids,
            'is_random_next': False,
            'masked_lm_positions': masked_lm_positions,
            'masked_lm_labels': masked_lm_labels
        }
        tmp_instance.append(instance)

        negs = []
        assert len(sel_answer_pool) > 10
        random_pool = random.sample(sel_answer_pool, k=10)
        for s in random_pool:
            if len(negs) > 1:
                break
            longest = SequenceMatcher(None, s, a).find_longest_match(0, len(s), 0, len(a))
            if longest.size / min(max(1,len(s)), max(1,len(a))) > 0.3 and len(a) > 5:
                continue
            negs.append(s)
        if len(negs) <=1:
            print('tst')
        for s in negs:
            tokens_b = [c for c in s]

            tokens = ["[CLS]"] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
            segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]
            tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(tokens)
            instance = {
                'tokens': tokens,
                'segment_ids': segment_ids,
                'is_random_next': True,
                'masked_lm_positions': masked_lm_positions,
                'masked_lm_labels': masked_lm_labels
            }
            tmp_instance.append(instance)

    return tmp_instance

if not os.path.exists(str(Path(data_dir)/'pretrain_data')):
    os.makedirs(str(Path(data_dir)/'pretrain_data'), exist_ok=True)

for epoch in range(3):
    epoch_filename = (Path(data_dir)/'pretrain_data'/f'epoch_{epoch}.json').open('w')
    instances = []
    with Pool(8) as p:
        func = partial(next_sentence)
        tmp_list = list(tqdm(p.imap(func, data, chunksize=64), desc=f'Epoch:{epoch}'))

        logger.info(f'tmp_list size: {len(tmp_list)}')
        for tmp in tmp_list:
            instances.extend(tmp)
        logger.info(f'instances size: {len(instances)}')

        for instance in instances:
            epoch_filename.write(json.dumps(instance, ensure_ascii=False) + '\n')

        metrics_filename = (Path(data_dir) / f'epoch_{epoch}_metrics.json').open('w')
        metrics_filename.write(json.dumps({'num_training_examples': len(instances)}))


