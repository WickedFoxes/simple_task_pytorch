import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.registry import register
from src.models.base import ModelBase

@register("model", "lstm_classifier")
class LSTMClassifier(ModelBase):
    def __init__(
            self, 
            vocab_size, 
            embed_dim, 
            hidden_dim, 
            num_layers, 
            bidirectional, 
            dropout, 
            pad_idx, 
            num_classes=2
        ):
        super().__init__()
        self.pad_idx = pad_idx
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # BiLSTM의 경우 hidden_dim * 2가 실제 출력 차원이 됨
        out_dim = hidden_dim * (2 if bidirectional else 1)

        # 최종 분류를 위한 FC 레이어
        self.fc = nn.Linear(out_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

        # Xavier 초기화
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(
            self, 
            input_ids, 
            **kwargs
        ):
        # input_ids: [B, T], lengths: [B]
        emb = self.dropout(self.embedding(input_ids))  # [B, T, E]
        lengths = (input_ids != self.pad_idx).sum(dim=1)

        # 패딩된 시퀀스를 압축
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # packed_output: 압축된 LSTM의 모든 시점의 은닉 상태
        # (h_n, c_n): 마지막 시점의 은닉 상태와 셀 상태
        packed_output, (h_n, c_n) = self.lstm(packed)
        
        # 압축을 풀어서 다시 패딩된 텐서로 변환
        # output: [B, T, H * num_directions]
        output, (h_n, c_n) = self.lstm(packed)  # h_n: [num_layers * num_directions, B, H]

        if self.bidirectional:
            # 마지막 layer의 forward/backward 은닉 상태 결합
            h_fwd = h_n[-2, :, :] # backword
            h_bwd = h_n[-1, :, :] # forward
            h = torch.cat([h_fwd, h_bwd], dim=1)  # [B, 2H]
        else:
            h = h_n[-1, :, :]  # [B, H]

        logits = self.fc(self.dropout(h))  # [B, C]
        return logits