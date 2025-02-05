import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import numpy as np

class SpatialPoseEstimator(nn.Module):
    def __init__(self, feature_dim=256, voxel_dims=(200, 200, 16)):
        super().__init__()
        self.feature_dim = feature_dim
        self.H, self.W, self.Z = voxel_dims
        
        # Define static and dynamic classes
        self.static_classes = {
            'terrain': 14,
            'manmade': 15,
            'driveable_surface': 11,
            'sidewalk': 13,
            'vegetation': 16
        }
        
        # Feature attention network
        self.attention_net = nn.Sequential(
            nn.Conv3d(feature_dim, 64, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # Feature refinement
        self.feature_refine = nn.Sequential(
            nn.Conv3d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm3d(feature_dim),
            nn.ReLU(),
            nn.Conv3d(feature_dim, feature_dim, 3, padding=1)
        )
        
        # Spatial encoding
        self.register_buffer('spatial_coords', self._init_spatial_coords())
        self.spatial_embed = nn.Sequential(
            nn.Conv3d(3, 32, 1),
            nn.ReLU(),
            nn.Conv3d(32, feature_dim, 1)
        )
        
        # Pose regression network
        self.pose_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 7)  # [tx, ty, tz, qx, qy, qz, qw]
        )

    def _init_spatial_coords(self):
        """Initialize normalized spatial coordinates"""
        x = torch.linspace(-1, 1, self.W)
        y = torch.linspace(-1, 1, self.H)
        z = torch.linspace(-1, 1, self.Z)
        
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
        coords = torch.stack([grid_x, grid_y, grid_z], dim=0)
        return coords

    def get_static_mask(self, occ):
        """Create mask for static regions from occupancy grid"""
        static_mask = torch.zeros_like(occ[..., 0])
        
        for cls_idx in self.static_classes.values():
            static_mask = torch.max(static_mask, occ[..., cls_idx])
            
        return static_mask

    def process_features(self, voxel_feat, occ):
        """Process features with spatial and semantic attention"""
        B = voxel_feat.size(0)
        
        # Get static regions mask
        static_mask = self.get_static_mask(occ)  # [B, H, W, Z]
        static_mask = static_mask.permute(0, 3, 1, 2).unsqueeze(1)  # [B, 1, Z, H, W]
        
        # Add spatial information
        spatial_feat = self.spatial_embed(
            self.spatial_coords.expand(B, -1, -1, -1, -1)
        )
        
        # Combine with geometric features
        voxel_feat = voxel_feat.permute(0, 4, 3, 1, 2)  # [B, F, Z, H, W]
        combined_feat = voxel_feat + spatial_feat
        
        # Refine features
        refined_feat = self.feature_refine(combined_feat)
        
        # Generate attention weights
        attention = self.attention_net(refined_feat)
        
        # Apply static mask and attention
        weighted_feat = refined_feat * attention * static_mask
        
        return weighted_feat, attention

    def aggregate_features(self, weighted_feat, attention):
        """Aggregate features with attention-based pooling"""
        # Global weighted pooling
        feat_sum = (weighted_feat * attention).sum(dim=(2,3,4))  # [B, F]
        attention_sum = attention.sum(dim=(2,3,4)) + 1e-6
        
        return feat_sum / attention_sum

    def match_features(self, feat1, feat2):
        """Match features between frames"""
        # Compute similarity matrix
        sim_matrix = torch.matmul(feat1, feat2.transpose(1, 2))
        
        # Normalize similarities
        matches = F.softmax(sim_matrix / 0.1, dim=-1)  # temperature = 0.1
        
        return matches

    def forward(self, prev_data, curr_data):
        """
        Estimate relative pose between frames
        Args:
            prev_data: dict with 'voxel_feat' [B,H,W,Z,F] and 'occ' [B,H,W,Z,C]
            curr_data: same as prev_data
        """
        # Process features for both frames
        prev_feat, prev_att = self.process_features(prev_data['voxel_feat'], 
                                                  prev_data['occ'])
        curr_feat, curr_att = self.process_features(curr_data['voxel_feat'], 
                                                  curr_data['occ'])
        
        # Aggregate features
        prev_global = self.aggregate_features(prev_feat, prev_att)
        curr_global = self.aggregate_features(curr_feat, curr_att)
        
        # Match features
        matches = self.match_features(curr_global.unsqueeze(1), 
                                    prev_global.unsqueeze(1))
        
        # Combine features for pose estimation
        matched_feat = torch.bmm(matches, prev_global.unsqueeze(1)).squeeze(1)
        combined = torch.cat([curr_global, matched_feat], dim=-1)
        
        # Estimate pose
        pose = self.pose_net(combined)
        translation = pose[:, :3]
        quaternion = F.normalize(pose[:, 3:], dim=-1)
        
        return {
            'translation': translation,
            'quaternion': quaternion,
            'attention': {
                'prev': prev_att,
                'curr': curr_att
            },
            'matches': matches,
            'static_weights': {
                'prev': self.get_static_mask(prev_data['occ']),
                'curr': self.get_static_mask(curr_data['occ'])
            }
        }

