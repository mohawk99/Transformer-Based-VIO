# fusion_transformer.py

import torch
import torch.nn as nn

class FusionTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(FusionTransformerEncoder, self).__init__()
        
        self.visual_pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        self.inertial_pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        self.modality_encoding = nn.Parameter(torch.randn(2, 1, d_model))  # Two modalities: visual and inertial


        self.visual_projection = nn.Linear(96000, d_model)
        self.inertial_projection = nn.Linear(512, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
    def forward(self, visual_tokens, inertial_tokens):

        visual_tokens = self.visual_projection(visual_tokens)
        inertial_tokens = self.inertial_projection(inertial_tokens)

        visual_tokens += self.visual_pos_encoding[:, :visual_tokens.size(1), :]
        inertial_tokens += self.inertial_pos_encoding[:, :inertial_tokens.size(1), :]
        
        visual_tokens += self.modality_encoding[0]
        inertial_tokens += self.modality_encoding[1]

        tokens = torch.cat((visual_tokens, inertial_tokens), dim=1)

        encoded_tokens = self.transformer_encoder(tokens)
        
        return encoded_tokens

class FusionTransformerDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1):
        super(FusionTransformerDecoder, self).__init__()
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.pose_output = nn.Linear(d_model, 7)
        self.target_projection = nn.Linear(1, d_model)   
        
    def forward(self, encoded_tokens, target):

        target_flat = target.view(-1, 1)  # (batch_size * sequence_length, input_dim)
        target_projected = self.target_projection(target_flat)  # shape: (14, 256)

        # Reshape it back to (2, 7, 256)
        target_projected = target_projected.view(2, 7, 256)

        target_mask = self._generate_square_subsequent_mask(target_projected.size(1)).to(target.device)

        target_projected = target_projected.permute(1,0,2)
        encoded_tokens = encoded_tokens.permute(1,0,2)

        print(f"Target shape : {target_projected.shape}")
        print(f"Encoded tokens shape : {encoded_tokens.shape}")

        decoded_tokens = self.transformer_decoder(target_projected, encoded_tokens, tgt_mask=target_mask)

        pose = self.pose_output(decoded_tokens)
        
        return pose
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class FusionTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1):
        super(FusionTransformer, self).__init__()
        self.encoder = FusionTransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.decoder = FusionTransformerDecoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout)
        
    def forward(self, visual_tokens, inertial_tokens, target):
        encoded_tokens = self.encoder(visual_tokens, inertial_tokens)
        pose = self.decoder(encoded_tokens, target)
        return pose
