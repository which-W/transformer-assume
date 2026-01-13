import torch
from torch.utils.data import Dataset
from collections import Counter
from typing import Iterable, List, Tuple

import spacy
de_nlp = spacy.load("de_core_news_sm")
en_nlp = spacy.load("en_core_web_sm")
#分词器准备
def de_tokenizer(text):
    return [tok.text.lower() for tok in de_nlp(text)]

def en_tokenizer(text):
    return [tok.text.lower() for tok in en_nlp(text)]
#读取相关的数据集
class Multi30kDataset(Dataset):
    def __init__(self, de_path: str, en_path: str):
        with open(de_path, encoding="utf-8") as f:
            self.de_lines = f.readlines()
        with open(en_path, encoding="utf-8") as f:
            self.en_lines = f.readlines()

        assert len(self.de_lines) == len(self.en_lines)

    def __len__(self):
        return len(self.de_lines)

    def __getitem__(self, idx):
        return self.de_lines[idx].strip(), self.en_lines[idx].strip()
#生成词表
class Vocab:
    def __init__(self, counter, specials):
        self.itos = list(specials)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        for token, freq in counter.items():
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

        self.default_index = None

    def set_default_index(self, idx):
        self.default_index = idx

    def __len__(self):
        return len(self.itos)

    def __call__(self, tokens: List[str]):
        return [
            self.stoi.get(tok, self.default_index)
            for tok in tokens
        ]


def build_vocab_from_iterator_compat(
    iterator: Iterable[List[str]],
    specials: List[str],
    special_first: bool = True,
):
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    return Vocab(counter, specials)

# ====== 特殊 token ======
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM = "<unk>", "<pad>", "<bos>", "<eos>"
# ====== dataset ======
DATA_DIR = "./multi30k"
train_dataset = Multi30kDataset(
    f"{DATA_DIR}/train.1.de",
    f"{DATA_DIR}/train.1.en",
)

# ====== 构建 token 序列 ======
de_tokens = []
en_tokens = []

for de, en in train_dataset:
    de_tokens.append(de_tokenizer(de))
    en_tokens.append(en_tokenizer(en))

# ====== vocab（接口与原来一致） ======
de_vocab = build_vocab_from_iterator_compat(
    de_tokens,
    specials=[UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM],
    special_first=True,
)
de_vocab.set_default_index(UNK_IDX)

en_vocab = build_vocab_from_iterator_compat(
    en_tokens,
    specials=[UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM],
    special_first=True,
)
en_vocab.set_default_index(UNK_IDX)

def de_preprocess(de_sentence):
    tokens = de_tokenizer(de_sentence)
    tokens = [BOS_SYM] + tokens + [EOS_SYM]
    ids = de_vocab(tokens)
    return tokens, ids


def en_preprocess(en_sentence):
    tokens = en_tokenizer(en_sentence)
    tokens = [BOS_SYM] + tokens + [EOS_SYM]
    ids = en_vocab(tokens)
    return tokens, ids

if __name__ == '__main__':
    # 词表大小
    print('de vocab:', len(de_vocab))
    print('en vocab:', len(en_vocab))

    # 特征预处理
    de_sentence,en_sentence=train_dataset[0]
    print('de preprocess:',*de_preprocess(de_sentence))
    print('en preprocess:',*en_preprocess(en_sentence))