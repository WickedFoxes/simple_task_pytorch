import os
import re
from typing import Any, Dict, Optional, Callable, Tuple
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
import sentencepiece as spm
import torch.nn.functional as F

from src.datasets.base import DatasetBase
from src.registry import register
    

class WMT16_DE_EN_Wrap(DatasetBase):
    def __init__(
        self,
        root: str = "./data",
        tokenizer=None,
        train: bool = True,
        max_len: int = 256,
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

        en_encoded = self.tokenizer.encode(en, out_type=int)
        de_encoded = self.tokenizer.encode(de, out_type=int, add_bos=True, add_eos=True)
        en_ids = torch.tensor(en_encoded, dtype=torch.long) # 다중클래스를 위한 정수 타입
        de_ids = torch.tensor(de_encoded, dtype=torch.long) # 다중클래스를 위한 정수 타입
        
        # en 길이제한 (Truncate + Pad)
        if len(en_ids) > self.max_len:
            en_ids = en_ids[:self.max_len]
        else:
            pad_len = self.max_len - len(en_ids)
            en_ids = F.pad(en_ids, (0, pad_len), value=self.pad_id)

        # de 길이제한 (Truncate + EOS + Pad)
        if len(de_ids) >= self.max_len:
            # 잘라내되 마지막은 EOS로 강제 설정
            de_ids = de_ids[:self.max_len]
            de_ids[-1] = self.eos_id
        else:
            # EOS 이후 PAD 채우기
            pad_len = self.max_len - len(de_ids)
            de_ids = F.pad(de_ids, (0, pad_len), value=self.pad_id)

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
            train=train,
        )

def _make_wmt16_collate_fn(pad_id, bos_id):
    def collate_fn(batch):
        src_list, tgt_list = [], []
        for src, tgt in batch:
            src_list.append(src)
            tgt_list.append(tgt)

        src_list_padded = pad_sequence(src_list, batch_first=True, padding_value=pad_id)
        tgt_list_padded = pad_sequence(tgt_list, batch_first=True, padding_value=pad_id)

        tgt_in = []
        tgt_out = []
        for t in tgt_list_padded:
            if len(t) == 0 or t[0] != bos_id:
                t = [bos_id] + list(t)
            # pad를 고려해 동일 길이로 만들기
            tgt_in.append(t[:-1])
            tgt_out.append(t[1:])
        
        tgt_in_padded = pad_sequence(tgt_in, batch_first=True, padding_value=pad_id)    # (B, T-1) → 길이 맞추기
        tgt_out_padded = pad_sequence(tgt_out, batch_first=True, padding_value=pad_id)  # (B, T-1)

        return src_list_padded, tgt_in_padded, tgt_out_padded
    return collate_fn


@register("dataset", "wmt16_de_en")
def build_wmt16_dataloaders(cfg: Dict[str, Any], **kwargs) -> Tuple[DataLoader, DataLoader]:
    # tokenizer = AutoTokenizer.from_pretrained(cfg.get("pretrained_tokenizer_name", "Helsinki-NLP/opus-mt-en-de"))
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(cfg["tokenizer_dir"])
    train_set = WMT16_DE_EN_Wrap.from_config(cfg, tokenizer=tokenizer, train=True)
    val_set   = WMT16_DE_EN_Wrap.from_config(cfg, tokenizer=tokenizer, train=False)

    # train 데이터 길이 제한
    train_max_len = cfg.get("max_train_data_len", None)
    valid_max_len = cfg.get("max_valid_data_len", None)
    if train_max_len is not None and train_max_len < len(train_set):
        train_set = Subset(train_set, range(train_max_len))
    if valid_max_len is not None and valid_max_len < len(val_set):
        val_set = Subset(val_set, range(valid_max_len))

    pad_id = cfg["pad_idx"]
    bos_id = cfg["bos_idx"]

    collate_fn = _make_wmt16_collate_fn(pad_id, bos_id)

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

