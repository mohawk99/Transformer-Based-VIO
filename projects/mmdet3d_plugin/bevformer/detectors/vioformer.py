
#import gc
#gc.collect()
import torch
torch.cuda.empty_cache()
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import copy
import numpy as np
from projects.mmdet3d_plugin.bevformer.detectors.pano_occ import PanoOcc
from ..modules.IMUTransformerEncoder import IMUTransformerEncoder
from ..modules.fusion_transformer import FusionTransformer
from ..modules.getgtposes import getvoposes,getimuposes
from ..dense_heads.pose_head import PoseNet

from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence

from ..modules.getgtposes import getvoposes
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='/content/drive/My Drive/Thesis/PanoOcc/data/occ3d-nus/', verbose=True)

@DETECTORS.register_module()
class VIOFormer(PanoOcc):
    def __init__(self,
                 imu_encoder,
                 fusion_transformer,
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
        super(VIOFormer, self).__init__(
            use_grid_mask, pts_voxel_layer, pts_voxel_encoder,
            pts_middle_encoder, pts_fusion_layer, img_backbone, pts_backbone,
            img_neck, pts_neck, pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained, video_test_mode)
        
        self.transformer_dim = 256

        self.vo_input_dim = 1280

        self.vo_pose_head = nn.Sequential(
            nn.LayerNorm(self.vo_input_dim),
            nn.Linear(self.vo_input_dim, self.vo_input_dim // 2),
            nn.GELU(),
            nn.Linear(self.vo_input_dim // 2, self.vo_input_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.vo_input_dim // 4, 7)  # 3 for translation, 4 for rotation
        )


        self.imu_pose_head = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim, self.transformer_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer_dim // 2, 10)  # 3 for position, 4 for orientation and 3 for velocity
        )


        # Initialize IMU encoder
        self.imu_encoder = IMUTransformerEncoder(imu_encoder)

        self.fusion_transformer = FusionTransformer(**fusion_transformer)
        

    @auto_fp16(apply_to=('img', 'imu_data'))
    def extract_feat(self, img, img_metas=None, len_queue=None, imu_data=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats
    

    @auto_fp16(apply_to=('img', 'imu_data', 'prev_bev'))

    def pose_loss(self, pose_preds, gt_poses):
        return torch.nn.MSELoss()(pose_preds, gt_poses)


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
                      imu_data=None,
                      prev_bev=None):
        

        losses = dict()

        # Extract features from images
        num_sequences = img.size(1)
        print("Num of sequences:", num_sequences)

        all_pred_poses = []
        stacked_multi_scale_feats = []

        for sequence_idx in range(num_sequences):
          current_img = img[:, sequence_idx, :, :, :, :]
          
          img_feats = self.extract_feat(current_img)
          
          scale_pooled_feats = []
          for scale_feat in img_feats:  
              B, T, C, H, W = scale_feat.shape
              scale_feat_flat = scale_feat.view(B, T, C, -1).mean(dim=-1)
              scale_pooled_feats.append(scale_feat_flat)
              stacked_multi_scale_feats.append(scale_pooled_feats)
          
          multi_scale_feats = torch.cat(scale_pooled_feats, dim=-1)
          
          
          aggregated_feats = multi_scale_feats.mean(dim=1)

          pose_output = self.vo_pose_head(aggregated_feats)
          all_pred_poses.append(pose_output)

        VO_poses = torch.stack(all_pred_poses, dim=1).view(-1, 7)
        print("VO poses shape:", VO_poses.shape)

        gt_poses = getvoposes(nusc, img_metas)  # Shape: [B * T, 7]
        device = VO_poses.device
        gt_poses = gt_poses.to(device)
        print("GT poses shape:", gt_poses.shape)


        #Extract features from IMU data
        batch_imu_data = []
        for i, img_meta_dict in enumerate(img_metas):
          for key, img_meta in img_meta_dict.items():
            can_bus_data = img_meta['can_bus_imu']
            print("can bus data:",can_bus_data)
            batch_imu_data.append(can_bus_data)

        batch_imu_data = torch.tensor(batch_imu_data, dtype=torch.float32).to(img.device)

        if batch_imu_data.dim() == 2:
            batch_imu_data = batch_imu_data.unsqueeze(1)
        elif batch_imu_data.dim() != 3:
            raise ValueError(f"Unexpected tensor shape: {batch_imu_data.shape}")
        
        imu_feats = self.imu_encoder({'data': batch_imu_data})
        print("IMU feats shape:", imu_feats.shape)
        imu_data =self.imu_pose_head(imu_feats)
        imu_data = imu_data.squeeze(1)
        imu_poses = imu_data[:, :7]
        imu_vel = imu_data[:, -3:]
        print("IMU poses shape:", imu_poses.shape) 


        ##Fusion Transformer
        # target_sequence = torch.zeros_like(gt_poses)  
        # fused_pose = self.fusion_transformer(stacked_multi_scale_feats, imu_feats, target_sequence)

        # print("Fused pose shape:", fused_pose.shape)

        # losses.update({'fused_pose_loss': self.pose_loss(fused_pose, gt_poses)})
        # print(f"losses : {losses}")
        # return losses

        fused_poses = []
        for sequence_idx in range(num_sequences):
            target_sequence = gt_poses[sequence_idx::num_sequences]
            fused_pose = self.fusion_transformer(
                stacked_multi_scale_feats[sequence_idx],  # Visual tokens for this sequence
                imu_feats[sequence_idx].unsqueeze(0),  # IMU tokens for this sequence
                target_sequence  # Target sequence for transformer
            )
            print("Fused pose shape:", fused_pose.shape)
            fused_poses.append(fused_pose)

        fused_poses = torch.cat(fused_poses, dim=0)  # Combine all fused poses
        print("Fused pose shape:", fused_poses.shape)

        # Calculate losses
        losses.update({'fused_pose_loss': self.pose_loss(fused_poses, gt_poses)})
        print(f"losses : {losses}")
        return losses


        
