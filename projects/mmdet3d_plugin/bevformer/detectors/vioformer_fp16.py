# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import copy
import numpy as np
from projects.mmdet3d_plugin.bevformer.detectors.bevformer_fp16 import BEVFormer_fp16
from projects.mmdet3d_plugin.bevformer.detectors.bevformer import BEVFormer
from ..modules.IMUTransformerEncoder import IMUTransformerEncoder
from ..modules.fusion_transformer import FusionTransformer
from ..modules.getgtposes import getgtposes
from ..dense_heads.pose_head import PoseNet

@DETECTORS.register_module()
class VIOFormer_fp16(BEVFormer):
    def __init__(self,
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
                 video_test_mode=False,
                 imu_encoder=None,
                 pose_net=None,
                 fusion_transformer=None):
        super(VIOFormer_fp16, self).__init__(
            use_grid_mask, pts_voxel_layer, pts_voxel_encoder,
            pts_middle_encoder, pts_fusion_layer, img_backbone, pts_backbone,
            img_neck, pts_neck, pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained, video_test_mode)

        # Initialize IMU encoder
        self.imu_encoder = IMUTransformerEncoder(imu_encoder) if imu_encoder else IMUTransformerEncoder(config.imu_encoder)

        self.fusion_transformer = FusionTransformer(**fusion_transformer) if fusion_transformer else FusionTransformer(**config.fusion_transformer)
        
        # Initialize PoseNet
        #self.pose_net = PoseNet(**pose_net) if pose_net else PoseNet(**config.pose_net)
        if pose_net:
            pose_net_cfg = {k: v for k, v in pose_net.items() if k != 'type'}
            self.pose_net = PoseNet(**pose_net_cfg)
        else:
            pose_net_cfg = {k: v for k, v in config.pose_net.items() if k != 'type'}
            self.pose_net = PoseNet(**pose_net_cfg)


    @auto_fp16(apply_to=('img', 'imu_data'))
    def extract_feat(self, img, img_metas=None, len_queue=None, imu_data=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats
    

    @auto_fp16(apply_to=('img', 'imu_data', 'prev_bev'))
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
        

        # Extract features from images
        img_feats = self.extract_feat(img=img, img_metas=img_metas, imu_data=imu_data)

        #img_feats_tensor = torch.tensor(img_feats, dtype=torch.float32).to(img.device)
        
        losses = dict()

        # Extract IMU data from can_bus for all samples in the batch
        batch_imu_data = []
        for meta in img_metas:
            can_bus_data = meta['can_bus']
            imu_data = can_bus_data[7:13]  # Assuming IMU data is within these indices
            batch_imu_data.append(imu_data)

        gt_poses = []
        for meta in img_metas:
            sample_idx=meta['sample_idx']
            gt_pose=getgtposes(sample_idx)
            gt_poses.append(gt_pose)

        gt_poses_tensor = torch.tensor(gt_poses, dtype=torch.float16).to(img.device)
    
        # Convert list of IMU data to tensor
        batch_imu_data = torch.tensor(batch_imu_data, dtype=torch.float16).to(img.device)
        #print(f"imu data length: {batch_imu_data.shape}")
        #print(f"imu data: {batch_imu_data}")

        if batch_imu_data.dim() == 2:
            batch_imu_data = batch_imu_data.unsqueeze(1)  # Add sequence dimension
        elif batch_imu_data.dim() != 3:
            raise ValueError(f"Unexpected tensor shape: {batch_imu_data.shape}")
        
        
        # Process the IMU data using the IMU encoder
        imu_feats = self.imu_encoder({'data': batch_imu_data})


        if isinstance(img_feats, list) and len(img_feats) == 1:
                img_feats = img_feats[0]  
        B, t, C, H, W = img_feats.shape
        img_feats = img_feats.view(B, t, -1)



        # Pose net fusion
        """ if self.pose_net:
            pose_preds = self.pose_net(img_feats, imu_feats)

            pose_preds = pose_preds.squeeze(1)
            
            losses.update({'pose_loss': self.pose_loss(pose_preds, gt_poses_tensor)})
            
        #return list(losses.values())
        print(f"losses : {losses}")
        return losses """



        #Transformer fusion 
        if self.fusion_transformer:
            target_sequence = torch.zeros_like(gt_poses_tensor)  
            pose_preds = self.fusion_transformer(img_feats, imu_feats, target_sequence)

            print(f"Pose preds : {pose_preds.shape}")
            print(f"GT pose : {gt_poses_tensor.shape}")

            pose_preds = pose_preds.mean(dim=0) # Because shape of pose preds is (7,2,7)

            losses.update({'pose_loss': self.pose_loss(pose_preds, gt_poses_tensor)})
            print(f"losses : {losses}")
            return losses
        
    def pose_loss(self, pose_preds, gt_poses):
        return torch.nn.MSELoss()(pose_preds, gt_poses)
