import torch
from torch.utils.data import Dataset



class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
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