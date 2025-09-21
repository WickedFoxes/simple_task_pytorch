import os
import re
from typing import Any, Dict, Optional, Callable, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from transformers import AutoTokenizer

from src.datasets.base import DatasetBase
from src.registry import register

def clean_text(text: str) -> str:
    # HTML 태그/줄바꿈 제거, 공백 정리
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

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
        token_ids = torch.tensor(encoded["input_ids"], dtype=torch.long) # 다중클래스를 위한 정수 타입
        # 길이 (pad 제외)
        length = sum(1 for t in encoded["input_ids"] if t != self.tokenizer.pad_token_id)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return token_ids, length, label
    

class IMDBWrap(DatasetBase):
    """
    HF datasets로 저장된 IMDB를 로드하고
    내부에서 ReviewDataset으로 감싼 래퍼.
    """
    def __init__(
        self,
        data_dir: str = "./data",
        tokenizer=None,
        max_len: int = 256,
        train: bool = True,
    ):
        super().__init__()
        # load HF dataset (train split만 저장되어 있다고 가정)
        self.tokenizer = tokenizer
        self.max_len = max_len
        # ReviewDataset은 (dataset_split, tokenizer, max_len) 시그니처로 가정
        if train:
            data = load_from_disk(os.path.join(data_dir, "train"))
        else:
            data = load_from_disk(os.path.join(data_dir, "test"))
        self.dataset = ReviewDataset(data, tokenizer, max_len)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # ReviewDataset이 (ids_tensor, length_int, label_tensor) 튜플을 반환한다고 가정
        return self.dataset[idx]

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        tokenizer=None,
        train: bool = True,
    ):
        return cls(
            data_dir=cfg.get("data_dir", "./data"),
            tokenizer=tokenizer,
            max_len=int(cfg.get("max_len", 256)),
            train=train,
        )

def _make_imdb_collate_fn(tokenizer):
    # pad token id 결정: 우선 tokenizer.pad_token_id, 없으면 vocab의 [PAD], 그마저도 없으면 0
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = tokenizer.get_vocab().get("[PAD]", 0)

    def collate_fn(batch):
        ids_list, lengths, labels = [], [], []
        for ids, length, label in batch:
            ids_list.append(ids)
            lengths.append(length)
            labels.append(label)

        padded = pad_sequence(ids_list, batch_first=True, padding_value=pad_id)
        lengths = torch.tensor(lengths, dtype=torch.long)
        labels = torch.stack(labels) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels, dtype=torch.long)
        return padded, lengths, labels

    return collate_fn

@register("dataset", "imdb")
def build_imdb_dataloaders(cfg: Dict[str, Any], **kwargs) -> Tuple[DataLoader, DataLoader]:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_set = IMDBWrap.from_config(cfg, tokenizer=tokenizer, train=True)
    val_set   = IMDBWrap.from_config(cfg, tokenizer=tokenizer, train=False)

    collate_fn = _make_imdb_collate_fn(tokenizer)

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg.get("batch_size", 128)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        val_set,
        batch_size=int(cfg.get("batch_size", 128)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader
