import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry import register
from src.models.base import ModelBase

from transformers import AutoModel, AutoConfig

@register("model", "bert_classifier")
class BertClassifier(ModelBase):
    def __init__(
            self, 
            num_classes: int=2, 
            pad_idx: int = 0,
            pooling: str = "cls",
            dropout:float=0.1,
            pretrained_model_name: str = "google-bert/bert-base-cased",
        ):
        super(BertClassifier, self).__init__()
        # 사전학습된 BERT 불러오기
        config = AutoConfig.from_pretrained(pretrained_model_name)
        self.bert = AutoModel.from_config(config)

        self.pooling = pooling.lower()
        self.pad_idx = pad_idx

        # BERT hidden size -- D : 768
        bert_hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_hidden_size, num_classes),
        )

    def _cls_pool(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        # [B, L, H] -> [B, H]
        return last_hidden_state[:, 0, :]

    def _mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # padding을 제외한 토큰만 평균
        # last_hidden_state: [B, L, H]
        # attention_mask:    [B, L] (1=유효 토큰, 0=패딩)
        if attention_mask is None:
            # 마스크가 없는 경우, 전 토큰 평균
            return last_hidden_state.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, L, 1]
        masked_sum = (last_hidden_state * mask).sum(dim=1)               # [B, H]
        lengths = mask.sum(dim=1).clamp(min=1e-6)                        # [B, 1]
        return masked_sum / lengths

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        attention_mask = (input_ids != self.pad_idx).long()

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = outputs.last_hidden_state  # [B, L, H]
        
        if self.pooling == "cls":
            pooled = self._cls_pool(last_hidden_state)
        else:  # "mean"
            pooled = self._mean_pool(last_hidden_state, attention_mask)

        logits = self.classifier(pooled)  # [B, num_classes]
        return logits


