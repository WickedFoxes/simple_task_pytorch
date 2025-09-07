
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, bidirectional, dropout, pad_idx):
        super().__init__()
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
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)
        self.dropout = nn.Dropout(dropout)

        # Xavier 초기화(선택)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, input_ids, lengths):
        # input_ids: [B, T], lengths: [B]
        emb = self.dropout(self.embedding(input_ids))  # [B, T, E]
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.lstm(packed)  # h_n: [num_layers * num_directions, B, H]

        if self.bidirectional:
            # 마지막 layer의 forward/backward 은닉 상태 결합
            h_fwd = h_n[-2, :, :] # backword
            h_bwd = h_n[-1, :, :] # forward
            h = torch.cat([h_fwd, h_bwd], dim=1)  # [B, 2H]
        else:
            h = h_n[-1, :, :]  # [B, H]

        logits = self.fc(self.dropout(h)).squeeze(1)  # [B]
        return logits  # (BCEWithLogitsLoss와 함께 사용)