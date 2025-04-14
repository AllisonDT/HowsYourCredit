import torch
import torch.nn as nn

class TabularTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_heads=8, num_layers=4, dropout=0.1, num_classes=3):
        super(TabularTransformer, self).__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_head = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)     # (B, 1, model_dim)
        x = self.norm(x)                         # (B, 1, model_dim)
        transformed = self.transformer_encoder(x)  # (B, 1, model_dim)
        x = x + transformed                      # Residual connection
        x = x.squeeze(1)                         # (B, model_dim)
        return self.cls_head(x)
