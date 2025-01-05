import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusion(nn.Module):
    """Cross-scale attention mechanism for multi-scale feature fusion."""
    def __init__(self, d_model, num_scales):
        super().__init__()
        self.input_proj = nn.Linear(d_model * num_scales, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=8)
        self.proj = nn.Linear(d_model, d_model)  # Reduce concatenated features to d_model

    def forward(self, scale_features):
        concatenated_features = torch.cat(scale_features, dim=-1)  # Shape: (B, T, d_model * num_scales)
        projected_features = self.input_proj(concatenated_features)  # Shape: (B, T, d_model)
        aggregated_features, _ = self.attn(projected_features, projected_features, projected_features)

        return self.proj(aggregated_features)


class CrossModalityAttention(nn.Module):
    """Cross-attention mechanism for visual and inertial token fusion."""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.cross_attn_visual_to_inertial = nn.MultiheadAttention(d_model, nhead)
        self.cross_attn_inertial_to_visual = nn.MultiheadAttention(d_model, nhead)

    def forward(self, visual_tokens, inertial_tokens):
        # Cross-attention: visual attends to inertial
        fused_visual, _ = self.cross_attn_visual_to_inertial(visual_tokens, inertial_tokens, inertial_tokens)
        # Cross-attention: inertial attends to visual
        fused_inertial, _ = self.cross_attn_inertial_to_visual(inertial_tokens, visual_tokens, visual_tokens)
        return fused_visual, fused_inertial


class PoseRegressionHead(nn.Module):
    """Pose regression head for 7-DOF pose prediction."""
    def __init__(self, d_model):
        super().__init__()
        self.fc_translation = nn.Linear(d_model, 3)  # Predict 3D translation
        self.fc_rotation = nn.Linear(d_model, 4)    # Predict quaternion

    def forward(self, tokens):
        translation = self.fc_translation(tokens)
        rotation = self.fc_rotation(tokens)
        rotation = F.normalize(rotation, dim=-1)  # Normalize quaternion
        return torch.cat([translation, rotation], dim=-1)


class RelativeMultiheadAttention(nn.MultiheadAttention):
    """Multi-head attention with relative positional encoding."""
    def __init__(self, embed_dim, num_heads, max_len=5000, **kwargs):
        super().__init__(embed_dim, num_heads, **kwargs)
        self.relative_pos_embedding = nn.Parameter(torch.randn(max_len, embed_dim))

    def forward(self, query, key, value, **kwargs):
        seq_len = query.size(0)
        relative_pos = self.relative_pos_embedding[:seq_len]
        query += relative_pos
        key += relative_pos
        return super().forward(query, key, value, **kwargs)


class FusionTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, num_scales):
        super().__init__()
        self.visual_pos_encoding = nn.Parameter(torch.randn(1, 500, d_model))  # Max sequence length = 500
        self.inertial_pos_encoding = nn.Parameter(torch.randn(1, 500, d_model))
        self.modality_encoding = nn.Parameter(torch.randn(2, 1, d_model))  # Two modalities: visual and inertial

        self.cross_scale_fusion = MultiScaleFusion(d_model, num_scales)

        # Projection layers for visual and inertial features
        self.visual_projection = nn.Linear(d_model * num_scales, d_model)  # After MultiScaleFusion
        self.inertial_projection = nn.Linear(6, d_model)  # Replace imu_input_dim with actual input size

        self.cross_modality_attention = CrossModalityAttention(d_model, nhead)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    def forward(self, img_feats, imu_feats):
        # Process multi-scale image features
        # scale_pooled_feats = []
        # for scale_feat in img_feats:  
        #     B, T, C, H, W = scale_feat.shape
        #     scale_feat_flat = scale_feat.view(B, T, C, -1).mean(dim=-1)  # Mean pooling over spatial dims
        #     scale_pooled_feats.append(scale_feat_flat)
            
        # print("pooled feats shape:",scale_pooled_feats.shape)
        # Cross-scale fusion
        visual_tokens = self.cross_scale_fusion(img_feats)  # Shape: (B, T, d_model)

        # Project visual features to d_model
        #visual_tokens = self.visual_projection(fused_visual_feats)  # Shape: (B, T, d_model)

        # Project IMU features to d_model
        #inertial_tokens = self.inertial_projection(imu_feats)  # Shape: (B, T, d_model)
        inertial_tokens = imu_feats

        # Add positional encodings
        visual_tokens += self.visual_pos_encoding[:, :visual_tokens.size(1), :]
        inertial_tokens += self.inertial_pos_encoding[:, :inertial_tokens.size(1), :]

        # Add modality encodings
        visual_tokens += self.modality_encoding[0]
        inertial_tokens += self.modality_encoding[1]

        print(f"Visual tokens shape: {visual_tokens.shape}") 
        print(f"Inertial tokens shape: {inertial_tokens.shape}")

        if inertial_tokens.size(1) < visual_tokens.size(1):
          inertial_tokens = inertial_tokens.expand(-1, visual_tokens.size(1), -1)

        # Cross-modality attention
        visual_tokens, inertial_tokens = self.cross_modality_attention(visual_tokens, inertial_tokens)

        # Concatenate tokens and pass through transformer encoder
        tokens = torch.cat((visual_tokens, inertial_tokens), dim=1)  # Shape: (B, T1 + T2, d_model)
        encoded_tokens = self.transformer_encoder(tokens)

        return encoded_tokens



class FusionTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.target_projection = nn.Linear(7, d_model)  # Project target to d_model
        self.pose_output = PoseRegressionHead(d_model)

    def forward(self, encoded_tokens, target):
        B, T = target.size()  # Expect target shape: (1, 7)

        # Project target sequence to d_model
        target_projected = self.target_projection(target)  # Shape: (B, d_model)
        target_projected = target_projected.unsqueeze(0)  # Shape: (1, B, d_model) for Transformer

        # Generate target mask (since we have only one timestep, this is trivial)
        target_mask = self._generate_square_subsequent_mask(1).to(target.device)

        # Decode tokens
        decoded_tokens = self.transformer_decoder(
            target_projected,  # Shape: (1, B, d_model)
            encoded_tokens.permute(1, 0, 2),  # Shape: (S, B, d_model)
            tgt_mask=target_mask
        )

        # Predict pose
        pose = self.pose_output(decoded_tokens.squeeze(0))  # Remove sequence dim, shape: (B, 7)
        return pose

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class FusionTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, num_scales=5):
        super().__init__()
        self.encoder = FusionTransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout, num_scales)
        self.decoder = FusionTransformerDecoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout)

    def forward(self, visual_tokens, inertial_tokens, target):
        # Encode tokens
        encoded_tokens = self.encoder(visual_tokens, inertial_tokens)

        # Decode tokens to predict pose
        pose = self.decoder(encoded_tokens, target)
        return pose
