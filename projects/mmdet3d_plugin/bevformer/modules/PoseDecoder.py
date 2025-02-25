import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix. Works with PyTorch tensors.
    Args:
        q: tensor of shape (..., 4) containing w,x,y,z
    Returns:
        tensor of shape (..., 3, 3)
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    r00 = 1 - 2*y*y - 2*z*z
    r01 = 2*x*y - 2*w*z
    r02 = 2*x*z + 2*w*y
    
    r10 = 2*x*y + 2*w*z
    r11 = 1 - 2*x*x - 2*z*z
    r12 = 2*y*z - 2*w*x
    
    r20 = 2*x*z - 2*w*y
    r21 = 2*y*z + 2*w*x
    r22 = 1 - 2*x*x - 2*y*y
    
    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1)
    ], dim=-2)
    
    return R

def quaternion_multiply(q1, q2):
    """Multiply two quaternions. Works with PyTorch tensors.
    Args:
        q1, q2: tensors of shape (..., 4) containing w,x,y,z
    Returns:
        tensor of shape (..., 4)
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack([w, x, y, z], dim=-1)

def quaternion_conjugate(q):
    """Compute quaternion conjugate. Works with PyTorch tensors.
    Args:
        q: tensor of shape (..., 4) containing w,x,y,z
    Returns:
        tensor of shape (..., 4)
    """
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


class BEVPoseEstimator(nn.Module):
    def __init__(self, max_shift=7, bev_h=50, bev_w=50):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.max_shift = max_shift
        
        # Process correlation volume 
        in_channels = (2*max_shift + 1)**2
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1)
        )
        
        # Pose head
        self.pose_head = nn.Sequential(
            nn.Linear(64*bev_h*bev_w, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # [x,y,z, qw,qx,qy,qz]
        )

    def shift_feature_map(self, x, dx, dy):
        """Shifts feature map by dx,dy pixels with zero padding"""
        if dx == 0 and dy == 0:
            return x
            
        N, C, H, W = x.shape
        shifted = torch.zeros_like(x)
        
        # Source indices
        src_h1 = max(0, dx)
        src_h2 = min(H, H + dx)
        src_w1 = max(0, dy)
        src_w2 = min(W, W + dy)
        
        # Target indices
        dst_h1 = max(0, -dx)
        dst_h2 = min(H, H - dx)
        dst_w1 = max(0, -dy)
        dst_w2 = min(W, W - dy)
        
        # Calculate valid height and width ranges
        h_valid = min(src_h2 - src_h1, dst_h2 - dst_h1)
        w_valid = min(src_w2 - src_w1, dst_w2 - dst_w1)
        
        if h_valid > 0 and w_valid > 0:
            shifted[..., dst_h1:dst_h1+h_valid, dst_w1:dst_w1+w_valid] = \
                x[..., src_h1:src_h1+h_valid, src_w1:src_w1+w_valid]
        
        return shifted

    def compute_correlation(self, bev1, bev2):
        """Compute correlation between BEV features
        Args:
            bev1, bev2: (N*H*W, 1, C) BEV features where N is batch size
        Returns:
            correlation: (N, 2*max_shift+1, 2*max_shift+1, H, W)
        """
        # First get the true batch size
        N = bev1.shape[0] // (self.bev_h * self.bev_w)
        C = 1
        
        # Reshape to (N, H, W, C)
        # bev1 = bev1.view(N, self.bev_h, self.bev_w, C)
        # bev2 = bev2.view(N, self.bev_h, self.bev_w, C)

        bev1 = bev1.view(N, self.bev_h, self.bev_w, -1)  # -1 preserves all channels
        bev2 = bev2.view(N, self.bev_h, self.bev_w, -1)
        
        # Permute to (N, C, H, W)
        bev1 = bev1.permute(0, 3, 1, 2)
        bev2 = bev2.permute(0, 3, 1, 2)
        
        corr_volume = torch.zeros(N, 2*self.max_shift+1, 2*self.max_shift+1, 
                                self.bev_h, self.bev_w, device=bev1.device)

        for dx in range(-self.max_shift, self.max_shift+1):
            for dy in range(-self.max_shift, self.max_shift+1):
                # Shift bev2
                shifted = self.shift_feature_map(bev2, dx, dy)
                
                # Compute correlation across all channels
                corr = torch.sum(bev1 * shifted, dim=1)
                
                # Store in volume
                corr_volume[:, dx+self.max_shift, dy+self.max_shift] = corr

        return corr_volume

    def forward(self, bev1, bev2):
        """
        Args:
            bev1, bev2: (N*H*W, 1, C) BEV features
        Returns:
            translation: (N, 3)
            quaternion: (N, 4)
        """
        # Get true batch size
        N = bev1.shape[0] // (self.bev_h * self.bev_w)
        
        # Compute correlation volume
        corr = self.compute_correlation(bev1, bev2)
        
        # Process correlation volume
        x = self.conv_layers(corr.reshape(N, -1, self.bev_h, self.bev_w))
        x = x.flatten(1)
        pose = self.pose_head(x)
        
        # Split and normalize outputs
        translation = pose[:, :3]
        quaternion = F.normalize(pose[:, 3:], dim=-1)
        
        return translation, quaternion