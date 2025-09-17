
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from typing import Tuple

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, bidirectional, dropout, pad_idx, num_classes=2):
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
        self.fc = nn.Linear(out_dim, num_classes)
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

        logits = self.fc(self.dropout(h))  # [B, C]
        return logits
    


# ----- Gate-level LayerNorm LSTM Cell -----
class LNLSTMCell(nn.Module):
    """
    LN(Wh h_{t-1}) + LN(Wx x_t) + b  -> [f, i, o, g]
    c_t = sigma(f) ⊙ c_{t-1} + sigma(i) ⊙ tanh(g)
    h_t = sigma(o) ⊙ tanh(LN(c_t))
    """
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True, dropconnect: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Separate affine for x_t and h_{t-1}; bias는 합산 후에 한 번만 사용
        self.x2h = nn.Linear(input_dim, 4 * hidden_dim, bias=False)
        
        # self.h2h = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        
        # ----- DropConnect 적용 부분 -----
        # 1. h2h 선형 레이어를 먼저 정의합니다.
        h2h_layer = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        # 2. WeightDrop으로 감싸서 순환 가중치('weight')에 드롭커넥트를 적용합니다.
        self.h2h = WeightDrop(h2h_layer, ['weight'], dropout=dropconnect)
        # ---------------------------------
        
        self.bias = nn.Parameter(torch.zeros(4 * hidden_dim)) if bias else None

        # LayerNorms for gate pre-activations and cell state
        self.ln_x = nn.LayerNorm(4 * hidden_dim)
        self.ln_h = nn.LayerNorm(4 * hidden_dim)
        self.ln_c = nn.LayerNorm(hidden_dim)

    def forward(self, x_t: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = hx  # [B,H], [B,H]

        gates = self.ln_x(self.x2h(x_t)) + self.ln_h(self.h2h(h_prev))
        if self.bias is not None:
            gates = gates + self.bias

        f_t, i_t, o_t, g_t = gates.chunk(4, dim=-1)
        f_t = torch.sigmoid(f_t)
        i_t = torch.sigmoid(i_t)
        o_t = torch.sigmoid(o_t)
        g_t = torch.tanh(g_t)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(self.ln_c(c_t))
        return h_t, c_t


# ----- Stacked (bi)directional Layer using LNLSTMCell -----
class LNLSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1,
                 bidirectional=False, dropout=0.0, dropconnect=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # forward cells per layer
        self.f_cells = nn.ModuleList([
            LNLSTMCell(input_dim if l == 0 else hidden_dim, hidden_dim, dropconnect=dropconnect)
            for l in range(num_layers)
        ])
        if bidirectional:
            # backward cells per layer
            self.b_cells = nn.ModuleList([
                LNLSTMCell(input_dim if l == 0 else hidden_dim, hidden_dim, dropconnect=dropconnect)
                for l in range(num_layers)
            ])

    @staticmethod
    def _reverse_padded_sequence(x, lengths):
        # x: [B,T,E], lengths: [B]
        B, T, E = x.size()
        x_rev = x.clone()
        for b in range(B):
            L = lengths[b].item()
            if L > 0:
                x_rev[b, :L] = torch.flip(x[b, :L], dims=[0])
        return x_rev

    def _run_direction(self, x, lengths, cells):
        """
        x: [B,T,E], lengths: [B]
        returns: seq_out [B,T,H], last_h [B,H]
        """
        device = x.device
        B, T, _ = x.shape
        seq = x

        for l, cell in enumerate(cells):
            h = torch.zeros(B, self.hidden_dim, device=device)
            c = torch.zeros(B, self.hidden_dim, device=device)
            outputs = []

            for t in range(T):
                h_new, c_new = cell(seq[:, t, :], (h, c))
                # mask: t < length인 배치만 업데이트
                mask = (t < lengths).float().unsqueeze(-1)  # [B,1]
                h = h_new * mask + h * (1 - mask)
                c = c_new * mask + c * (1 - mask)
                outputs.append(h)

            seq = torch.stack(outputs, dim=1)  # [B,T,H]
            if l < self.num_layers - 1:
                seq = self.dropout(seq)

        # 각 배치의 마지막 유효 타임스텝 인덱스로 last h 추출
        last_idx = (lengths - 1).clamp(min=0)
        last_h = seq[torch.arange(B, device=device), last_idx, :]  # [B,H]
        return seq, last_h

    def forward(self, x, lengths):
        # FORWARD
        f_seq, f_last = self._run_direction(x, lengths, self.f_cells)

        if not self.bidirectional:
            return f_seq, f_last

        # BACKWARD (유효 구간만 역순)
        x_rev = self._reverse_padded_sequence(x, lengths)
        b_seq_rev, b_last = self._run_direction(x_rev, lengths, self.b_cells)
        # 필요하면 시퀀스 자체를 다시 원래 순서로 복원할 수 있음:
        b_seq = self._reverse_padded_sequence(b_seq_rev, lengths)

        return (torch.cat([f_seq, b_seq], dim=-1),  # [B,T,2H]
                torch.cat([f_last, b_last], dim=-1))  # [B,2H]


# ----- 최종 분류기 -----
class LSTMClassifier_v2(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 bidirectional, dropout, pad_idx, num_classes=2, dropconnect=0.0):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.rnn = LNLSTMLayer(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
            dropconnect=dropconnect,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

        # (선택) 분류기 가중치 초기화
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, input_ids, lengths):
        # input_ids: [B,T], lengths: [B]
        emb = self.embedding(input_ids)          # [B,T,E]
        emb = self.dropout(emb)

        # 우리 구현은 pack 없이 lengths 마스킹으로 가변 길이 처리
        _, h_last = self.rnn(emb, lengths)       # [B,out_dim]
        logits = self.fc(self.dropout(h_last))   # [B,C]
        return logits


class WeightDrop(nn.Module):
    """
    기존 모듈의 가중치에 DropConnect를 적용하는 래퍼 클래스.
    forward pass마다 지정된 가중치에 dropout을 적용합니다.
    """
    def __init__(self, module: nn.Module, weights: list, dropout: float = 0.0):
        super().__init__()
        self.module = module
        self.weights = weights  # 드롭아웃을 적용할 가중치 파라미터의 이름 리스트 (예: ['weight'])
        self.dropout = dropout
        self._setup()

    def _setup(self):
        # 원본 가중치 파라미터를 '_raw' 접미사를 붙여 백업하고,
        # 모듈에서 해당 파라미터를 삭제합니다.
        # 이렇게 해야 원본 가중치가 nn.Module의 파라미터로 중복 등록되지 않습니다.
        for name in self.weights:
            param = getattr(self.module, name)
            delattr(self.module, name)
            self.register_parameter(name + '_raw', nn.Parameter(param.data))

    def _set_weights(self):
        # forward 시점에 '_raw' 가중치에 드롭아웃을 적용하여
        # 모듈의 원래 가중치 이름으로 설정합니다.
        for name in self.weights:
            raw_w = getattr(self, name + '_raw')
            # 훈련 모드일 때만 드롭아웃 적용
            w = F.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name, w)

    def forward(self, *args):
        self._set_weights()
        return self.module(*args)
