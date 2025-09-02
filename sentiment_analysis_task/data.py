import re

import torch
from torch.utils.data import Dataset



class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.unk_idx = self.vocab["<unk>"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(text)
        token_ids = [self.vocab[token] for token in tokens][: self.max_len]
        length = len(token_ids)
        if length == 0:
            # 빈 시퀀스 방지 (모두 OOV 등)
            token_ids = [self.unk_idx]
            length = 1
        label = float(self.labels[idx])
        return torch.tensor(token_ids, dtype=torch.long), length, torch.tensor(label, dtype=torch.float32)