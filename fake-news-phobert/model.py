import torch.nn as nn
from transformers import AutoModel


class TransformerClassifier(nn.Module):
    """Encoder (PhoBERT / XLM-R) + mean pooling + classification head."""

    def __init__(self, name="vinai/phobert-base", n_classes=2, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        h = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(h, h // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h // 2, n_classes),
        )

    def forward(self, input_ids, attention_mask, **_):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled)
