import os
import re

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from datasets import load_from_disk

def build_train_dataloader(
    dataset_name,
    tokenizer,
    data_dir,
    batch_size,
    num_workers=4,
    valid_ratio=0.1,
    max_len=400
):
    if dataset_name == 'imdb':
        train_loader, valid_loader = get_imdb_train_dataloader(
            tokenizer,
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            valid_ratio=valid_ratio,
            max_len=max_len
        )
    return train_loader, valid_loader

def build_test_dataloader(
    dataset_name,
    tokenizer,
    data_dir,
    batch_size,
    num_workers=4,
    max_len=400 
):
    if dataset_name == 'imdb':
        test_loader = get_imdb_test_dataloader(
            tokenizer,
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            max_len=max_len
        )
    return test_loader
    
def get_imdb_train_dataloader(
        tokenizer,
        data_dir="./data",
        batch_size=128,
        num_workers=4,
        max_len=256,
        valid_ratio=0.1,
    ):
    dataset = load_from_disk(os.path.join(data_dir, "train"))
    dataset = dataset.train_test_split(test_size=valid_ratio)
    
    train_ds = ReviewDataset(dataset["train"], tokenizer, max_len)
    valid_ds  = ReviewDataset(dataset["test"],  tokenizer, max_len)

    pad_idx = tokenizer.get_vocab()["[PAD]"]
    def collate_fn(batch):
        ids_list, lengths, labels = [], [], []
        for ids, length, label in batch:
            ids_list.append(ids)
            lengths.append(length)
            labels.append(label)
        
        padded = pad_sequence(ids_list, batch_first=True, padding_value=pad_idx)
        lengths = torch.tensor(lengths, dtype=torch.long)
        labels = torch.stack(labels)
        return padded, lengths, labels
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    return train_loader, valid_loader

def get_imdb_test_dataloader(
        tokenizer,
        data_dir="./data",
        batch_size=128,
        num_workers=4,
        max_len=256,
    ):
    dataset = load_from_disk(os.path.join(data_dir, "test"))
    test_ds = ReviewDataset(dataset, tokenizer, max_len)

    pad_idx = tokenizer.get_vocab()["[PAD]"]
    def collate_fn(batch):
        ids_list, lengths, labels = [], [], []
        for ids, length, label in batch:
            ids_list.append(ids)
            lengths.append(length)
            labels.append(label)
        
        padded = pad_sequence(ids_list, batch_first=True, padding_value=pad_idx)
        lengths = torch.tensor(lengths, dtype=torch.long)
        labels = torch.stack(labels)
        return padded, lengths, labels
    
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    return test_loader


class ReviewDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len=256):
        self.texts = dataset['text']
        self.labels = dataset['label']
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            clean_text(text),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",   # LSTM 입력을 동일 길이로 맞춤
            truncation=True
        )
        token_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        # 길이 (pad 제외)
        length = sum(1 for t in encoded["input_ids"] if t != self.tokenizer.pad_token_id)
        label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return token_ids, length, label
    

def clean_text(text: str) -> str:
    # HTML 태그/줄바꿈 제거, 공백 정리
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text