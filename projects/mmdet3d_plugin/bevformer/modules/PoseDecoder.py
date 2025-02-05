import torch
import torch.nn as nn
import torch.nn.functional as F

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
        bev1 = bev1.view(N, self.bev_h, self.bev_w, C)
        bev2 = bev2.view(N, self.bev_h, self.bev_w, C)
        
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