import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class IMUTransformerEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.transformer_dim = config.get("transformer_dim")

        self.input_proj = nn.Sequential(nn.Conv1d(config.get("input_dim"), self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU())

        self.window_size = config.get("window_size")
        self.encode_position = config.get("encode_position")
        encoder_layer = TransformerEncoderLayer(d_model=self.transformer_dim,
                                                nhead=config.get("nhead"),
                                                dim_feedforward=config.get("dim_feedforward"),
                                                dropout=config.get("transformer_dropout"),
                                                activation=config.get("transformer_activation"))

        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                                      num_layers=config.get("num_encoder_layers"),
                                                      norm=nn.LayerNorm(self.transformer_dim))

        if self.encode_position:
            #self.position_embed = nn.Parameter(torch.randn(self.window_size, 1, self.transformer_dim))
            self.position_embed = SinusoidalPositionalEncoding(self.transformer_dim, self.window_size)

        self.pose_head = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim, self.transformer_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer_dim // 2, 7)  # 3 for position and 4 for orientation
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        src_data = data['data'].transpose(1, 2)

        # Embed in a high dimensional space and reshape to Transformer's expected shape
        src = self.input_proj(src_data).permute(2, 0, 1)

        if self.encode_position:
            src += self.position_embed(src)

        # Pass through the Transformer encoder
        transformer_output = self.transformer_encoder(src)

        output = transformer_output.permute(1, 0, 2)

        # Apply the regression head to each time step
        #output = self.pose_head(transformer_output.permute(1, 0, 2))

        return output
