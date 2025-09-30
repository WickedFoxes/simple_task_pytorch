import os
import re
from typing import Any, Dict, Optional, Callable, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer

from src.datasets.base import DatasetBase
from src.registry import register
    

class WMT16_DE_EN_Wrap(DatasetBase):
    def __init__(
        self,
        root: str = "./data",
        tokenizer=None,
        max_len: int = 256,
        train: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        if train:
            self.ds = load_from_disk(os.path.join(root, "train"))
        else:
            self.ds = load_from_disk(os.path.join(root, "test"))

    def __len__(self):
        return len(self.ds["translation"])

    def __getitem__(self, idx):
        en = self.ds["translation"][idx]["en"]
        de = self.ds["translation"][idx]["de"]

        en_encoded = self.tokenizer(
            en,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",   # LSTM 입력을 동일 길이로 맞춤
            truncation=True
        )
        de_encoded = self.tokenizer(
            de,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",   # LSTM 입력을 동일 길이로 맞춤
            truncation=True
        )
        
        en_ids = torch.tensor(en_encoded["input_ids"], dtype=torch.long) # 다중클래스를 위한 정수 타입
        de_ids = torch.tensor(de_encoded["input_ids"], dtype=torch.long) # 다중클래스를 위한 정수 타입
        return en_ids, de_ids

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        tokenizer=None,
        train: bool = True,
    ):
        return cls(
            root=cfg.get("root", "./data"),
            tokenizer=tokenizer,
            max_len=int(cfg.get("max_len", 256)),
            train=train,
        )

def _make_wmt16_collate_fn(src_pad_id, tgt_pad_id, tgt_bos_id):
    def collate_fn(batch):
        src_list, tgt_list = [], []
        for src, tgt in batch:
            src_list.append(src)
            tgt_list.append(tgt)

        src_list_padded = pad_sequence(src_list, batch_first=True, padding_value=src_pad_id)
        tgt_list_padded = pad_sequence(tgt_list, batch_first=True, padding_value=tgt_pad_id)

        tgt_in = []
        tgt_out = []
        for t in tgt_list_padded:
            if len(t) == 0 or t[0] != tgt_bos_id:
                t = [tgt_bos_id] + list(t)
            # pad를 고려해 동일 길이로 만들기
            tgt_in.append(t[:-1])
            tgt_out.append(t[1:])
        
        tgt_in_padded = pad_sequence(tgt_in, tgt_pad_id)    # (B, T-1) → 길이 맞추기
        tgt_out_padded = pad_sequence(tgt_out, tgt_pad_id)  # (B, T-1)

        return src_list_padded, tgt_in_padded, tgt_out_padded
    return collate_fn

@register("dataset", "wmt16_de_en")
def build_wmt16_dataloaders(cfg: Dict[str, Any], **kwargs) -> Tuple[DataLoader, DataLoader]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.get("pretrained_tokenizer_name", "bert-base-uncased"))
    train_set = WMT16_DE_EN_Wrap.from_config(cfg, tokenizer=tokenizer, train=True)
    val_set   = WMT16_DE_EN_Wrap.from_config(cfg, tokenizer=tokenizer, train=False)

    pad_id = getattr(tokenizer, "pad_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)

    if pad_id is None:
        pad_id = tokenizer.get_vocab().get("<pad>", 1)
    if bos_id is None:
        bos_id = tokenizer.get_vocab().get("<s>", 0)

    collate_fn = _make_wmt16_collate_fn(pad_id, pad_id, bos_id)

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
