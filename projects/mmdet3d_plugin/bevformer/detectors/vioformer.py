
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
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/mohak/Thesis/PanoOcc/data/occ3d-nus/', verbose=True)

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
        
        self.transformer_dim = transformer_dim

        self.vo_pose_head = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim, self.transformer_dim // 2),
            nn.GELU(),
            nn.Linear(self.transformer_dim // 2, self.transformer_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer_dim // 4, 7)  # 3 for translation, 4 for rotation
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
        img_feats = self.extract_feat(img=img, img_metas=img_metas, imu_data=imu_data)
        num_sequences = img.size(1)
        all_pred_poses = []

        for sequence_idx in range(num_sequences):

            current_img = img[:, sequence_idx, :, :, :, :]
            img_feats = self.extract_feat(current_img)
            pred_poses = []
            B, T, C, H, W = img_feats[-1].shape
            img_feats_flat = img_feats[-1].view(B, T, C, -1).mean(dim=-1)
            img_feats_pooled = img_feats_flat.mean(dim=1)
            pose_output = self.pose_head(img_feats_pooled)
            pred_poses.append(pose_output)
            all_pred_poses = torch.stack(pred_poses, dim=1)

        VO_poses = all_pred_poses.view(-1, 7)
        print("VO poses shape:", VO_poses)

        gt_poses = getvoposes(nusc, img_metas)  # Shape: [B * T, 7]
        device = VO_poses.device
        gt_poses = gt_poses.to(device)


        

        #Extract features from IMU data
        batch_imu_data = []
        for meta in img_metas:
            can_bus_data = meta['can_bus_imu']
            batch_imu_data.append(can_bus_data)

        batch_imu_data = torch.tensor(batch_imu_data, dtype=torch.float16).to(img.device)

        if batch_imu_data.dim() == 2:
            batch_imu_data = batch_imu_data.unsqueeze(1)
        elif batch_imu_data.dim() != 3:
            raise ValueError(f"Unexpected tensor shape: {batch_imu_data.shape}")
        
        imu_feats = self.imu_encoder({'data': batch_imu_data})

        imu_data =self.imu_pose_head(imu_feats)
        imu_poses = imu_data[:7]
        imu_vel = imu_data[-3:]
        print("IMU feats shape:", imu_feats)
        print("IMU poses shape:", imu_poses)

        #Fusion Transformer
        target_sequence = torch.zeros_like(gt_poses)  
        fused_pose = self.fusion_transformer(img_feats, imu_feats, target_sequence)

        fused_pose = fused_pose.mean(dim=0) # Because shape of pose preds is (7,2,7)

        # losses.update({'fused_pose_loss': self.pose_loss(fused_pose, gt_poses)})
        # print(f"losses : {losses}")
        # return losses
        
