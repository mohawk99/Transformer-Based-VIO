import torch
import torch.nn as nn
import math
from util.IMU_NU import *

def force_quaternion_uniqueness(quat):
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)
    quat = torch.where(quat[..., 0:1] < 0, -quat, quat)
    return quat

class QuaternionLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.r = nn.Linear(in_features, out_features)
        self.i = nn.Linear(in_features, out_features)
        self.j = nn.Linear(in_features, out_features)
        self.k = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        qr = self.r(x)
        qi = self.i(x)
        qj = self.j(x)
        qk = self.k(x)
        return torch.cat([qr, qi, qj, qk], dim=-1)
    
class SensorProcessing(nn.Module):
    """Initial CNN processing for each sensor stream"""
    def __init__(self, in_channels=3, hidden_dim=256):
        super().__init__()
        self.process = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, hidden_dim//2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.GELU(),
        )
    
    def forward(self, x):
        return self.process(x)

class PositionalEncoding(nn.Module):
    """Learnable positional encoding"""
    def __init__(self, d_model, max_len=200):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        
    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]

class IMUTransformerEncoder(nn.Module):
    """Transformer encoder with separate processing for acc and gyro"""
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.get("hidden_dim")
        # Separate processing streams
        self.acc_process = SensorProcessing(in_channels=3, hidden_dim=self.hidden_dim)
        self.gyro_process = SensorProcessing(in_channels=3, hidden_dim=self.hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.get("nhead"),
            dim_feedforward=config.get("dim_feedforward"),
            dropout=config.get("dropout"),
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.get("num_encoder_layers")
        )
        
    def forward(self, acc_data, gyro_data):
        # Process each stream (B, C, L) -> (B, L, D/2)
        if acc_data.dim() == 3 and acc_data.size(1) > acc_data.size(2):
            acc_data = acc_data.transpose(1, 2)
        if gyro_data.dim() == 3 and gyro_data.size(1) > gyro_data.size(2):
            gyro_data = gyro_data.transpose(1, 2)

        acc_features = self.acc_process(acc_data).transpose(1, 2)
        gyro_features = self.gyro_process(gyro_data).transpose(1, 2)
        
        # Concatenate features (B, L, D)
        combined_features = torch.cat([acc_features, gyro_features], dim=-1)
        
        # Add positional encoding
        encoded_features = self.pos_encoding(combined_features)
        
        # Pass through transformer encoder
        memory = self.transformer_encoder(encoded_features)
        
        return memory

class IMUTransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.get("hidden_dim")
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=config.get("nhead"),
            dim_feedforward=config.get("dim_feedforward"),
            dropout=config.get("dropout"),
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.get("num_decoder_layers")
        )
        
        # Position head
        self.pos_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 3)
        )
        
        # Quaternion head
        self.quat_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 4),
            nn.Tanh()  # Constrain quaternion values
        )
        
    def forward(self, tgt, memory):
        decoder_output = self.transformer_decoder(tgt, memory)
        pos = self.pos_head(decoder_output)
        quat = self.quat_head(decoder_output)
        # quat = F.normalize(quat, p=2, dim=-1)  # Ensure unit quaternion
        quat = force_quaternion_uniqueness(quat)
        return torch.cat([pos, quat], dim=-1)
    


class IMUTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = IMUTransformerEncoder(config)
        self.decoder = IMUTransformerDecoder(config)
        self.query_embed = nn.Parameter(torch.randn(1, 1, config.get("hidden_dim")))
        
    def forward(self, acc_data, gyro_data):
        # Emphasize gyro data for better orientation learning
        if gyro_data.dim() == 3 and gyro_data.size(1) > gyro_data.size(2):
            gyro_data = gyro_data.transpose(1, 2)
        #gyro_data = gyro_data * 2.0  # Increase gyro influence
        
        memory = self.encoder(acc_data, gyro_data)
        batch_size = acc_data.size(0)
        query = self.query_embed.expand(batch_size, -1, -1)
        output = self.decoder(query, memory)
        return output

class IMUTransformerLoss(nn.Module):
   def __init__(self):
       super().__init__()
       #self.pos_weight = nn.Parameter(torch.ones(1) * 0.5)
       #self.ori_weight = nn.Parameter(torch.ones(1) * 1.0)
       self.pos_weight = nn.Parameter(torch.zeros(1))
       self.ori_weight = nn.Parameter(torch.zeros(1))
       
   def forward(self, pred, target):
       pred_pos = pred[..., :3] 
       pred_quat = pred[..., 3:7]
       target_pos = target[..., :3]
       target_quat = target[..., 3:7]
       
       device = pred.device
       pos_precision = torch.exp(-self.pos_weight.to(device))
       ori_precision = torch.exp(-self.ori_weight.to(device))
       
       #pos_loss = torch.abs(pred_pos - target_pos).mean()

       pred_pos = pred_pos.float()
       target_pos = target_pos.float()
       loss_fn = torch.nn.MSELoss()
       pos_loss = loss_fn(pred_pos, target_pos)


       quat_loss = 2 * torch.abs(quaternion_multiply(pred_quat, 
                                                   quaternion_conjugate(target_quat))).mean()
       
       total_loss = pos_precision * pos_loss + self.pos_weight.to(device) + ori_precision * quat_loss + self.ori_weight.to(device)
       #total_loss = pos_loss + quat_loss
       
       return total_loss, pos_loss, quat_loss



