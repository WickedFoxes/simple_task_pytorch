import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.registry import register
from src.models.base import ModelBase

@register("model", "lstm_attention_classifier")
class LSTMAttentionClassifier(ModelBase):
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
        
        # --- 어텐션 레이어 추가 ---
        # 어텐션 스코어를 계산하기 위한 레이어
        self.attention_layer = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.Tanh(),
        )
        # 어텐션 가중치를 계산하기 위한 쿼리 벡터 (학습 가능한 파라미터)
        self.attention_query = nn.Parameter(torch.randn(out_dim, 1))

        # 최종 분류를 위한 FC 레이어
        self.fc = nn.Linear(out_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

        # Xavier 초기화
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.attention_layer[0].weight)
        nn.init.xavier_uniform_(self.attention_query)

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
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # --- 어텐션 메커니즘 적용 ---
        # 1. 어텐션 스코어 계산
        # u: [B, T, out_dim]
        u = self.attention_layer(output)
        
        # attn_scores: [B, T, 1] -> [B, T]
        attn_scores = u.matmul(self.attention_query).squeeze(-1)

        # 2. 패딩 마스킹 (Padding Masking)
        # 실제 길이(lengths)를 벗어나는 패딩 토큰에는 어텐션을 주지 않기 위해
        # 매우 작은 값(-inf)을 할당하여 softmax 결과가 0이 되도록 함
        mask = torch.arange(input_ids.size(1))[None, :].to(input_ids.device) >= lengths[:, None]
        attn_scores.masked_fill_(mask, -float('inf'))

        # 3. 어텐션 가중치(확률 분포) 계산
        # attn_weights: [B, T]
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 4. 문맥 벡터(Context Vector) 계산
        # 가중치와 LSTM의 은닉 상태들을 가중합
        # attn_weights.unsqueeze(1): [B, 1, T]
        # output: [B, T, out_dim]
        # context_vector: [B, 1, out_dim] -> [B, out_dim]
        context_vector = torch.bmm(attn_weights.unsqueeze(1), output).squeeze(1)

        # 최종 분류
        logits = self.fc(self.dropout(context_vector))  # [B, C]
        return logits