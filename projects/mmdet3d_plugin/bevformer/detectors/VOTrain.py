import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
from torch import nn
from projects.mmdet3d_plugin.bevformer.detectors.pano_occ import PanoOcc
from ..modules.getgtposes import getvoposes
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/mohak/Thesis/PanoOcc/data/nuscenes', verbose=True)

class OpticalFlowHead(nn.Module):
    def __init__(self, in_channels):
        super(OpticalFlowHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 2, kernel_size=3, padding=1)

    def forward(self, feat1, feat2):
        flow_input = torch.cat([feat1, feat2], dim=1)
        x = F.relu(self.conv1(flow_input))
        x = F.relu(self.conv2(x))
        flow = self.conv3(x)
        return flow

def photometric_loss(img1, img2, flow):
    img2_warped = warp(img2, flow)
    loss = F.l1_loss(img1, img2_warped)
    return loss

def warp(img, flow):
    B, C, H, W = img.size()

    # Create a mesh grid representing the pixel indices
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # Shape: (H, W, 2)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # Shape: (B, H, W, 2)

    # Normalize grid values to [-1, 1]
    grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1  
    grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1  

    # Add optical flow to the grid
    flow = flow.permute(0, 2, 3, 1)  # Shape: (B, H, W, 2)
    flow_grid = grid + flow  # Add flow displacement

    # Clip the grid to be within the image bounds
    flow_grid[..., 0] = torch.clamp(flow_grid[..., 0], -1, 1)
    flow_grid[..., 1] = torch.clamp(flow_grid[..., 1], -1, 1)

    # Use grid_sample to warp the image
    warped_img = F.grid_sample(img, flow_grid, align_corners=True)

    return warped_img


@DETECTORS.register_module()
class VOTrain(PanoOcc):
    def __init__(self,
                 transformer_dim,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False):
        super(VOTrain, self).__init__(
            use_grid_mask, pts_voxel_layer, pts_voxel_encoder,
            pts_middle_encoder, pts_fusion_layer, img_backbone, pts_backbone,
            img_neck, pts_neck, pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained, video_test_mode)
        
        self.transformer_dim = transformer_dim

        self.pose_head = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim, self.transformer_dim // 2),
            nn.GELU(),
            nn.Linear(self.transformer_dim // 2, self.transformer_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer_dim // 4, 7)  # Output: 3 for translation, 4 for rotation (quaternion)
        )

        self.optical_flow_head = OpticalFlowHead(in_channels=self.transformer_dim // 4)

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    @auto_fp16(apply_to=('img', 'prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      prev_bev=None,
                      gt_poses=None):
        
        losses = dict()

        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        pred_poses = self.pose_head(img_feats[-1])

        gt_poses = getvoposes(nusc, img_metas)  

        pose_loss = F.mse_loss(pred_poses, gt_poses)
        losses['pose_loss'] = pose_loss

        feat1, feat2 = img_feats[-2], img_feats[-1]  
        flow_pred = self.optical_flow_head(feat1, feat2)

        optical_flow_loss = photometric_loss(img[:, -2], img[:, -1], flow_pred)
        losses['optical_flow_loss'] = optical_flow_loss

        return losses
