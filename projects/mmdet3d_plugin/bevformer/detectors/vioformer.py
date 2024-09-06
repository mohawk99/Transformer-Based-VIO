# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.bevformer.detectors.bevformer import BEVFormer
from projects.mmdet3d_plugin.bevformer.detectors.bevformer_fp16 import BEVFormer_fp16
from ..modules.IMUTransformerEncoder import IMUTransformerEncoder
from ..dense_heads.pose_head import *



print(DETECTORS)


@DETECTORS.register_module()
class VIOFormer(BEVFormer):
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
                 pose_net=None):
        super(VIOFormer, self).__init__(
            use_grid_mask, pts_voxel_layer, pts_voxel_encoder,
            pts_middle_encoder, pts_fusion_layer, img_backbone, pts_backbone,
            img_neck, pts_neck, pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained, video_test_mode)
        
        self.imu_encoder = imu_encoder if imu_encoder else IMUTransformerEncoder()
        self.pose_net = pose_net if pose_net else PoseNet()

    @auto_fp16(apply_to=('img', 'imu_data'))
    def extract_feat(self, img, img_metas=None, len_queue=None, imu_data=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats
    

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
                      imu_data=None): 

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas, imu_data=imu_data)
        
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)


        # Extract IMU data from can_bus for all samples in the batch
        batch_imu_data = []
        for meta in img_metas:
            can_bus_data = meta['can_bus']
            imu_data = can_bus_data[7:13]
            batch_imu_data.append(imu_data)
    
        # Convert list of IMU data to tensor
        batch_imu_data = torch.tensor(batch_imu_data).to(img.device)
        imu_feats = self.imu_encoder({'data': batch_imu_data})



        if self.pose_net:
            pose_preds = self.pose_net(img_feats, imu_feats)
            losses.update({'pose_loss': self.pose_loss(pose_preds, gt_poses)})

        return losses


    def pose_loss(self, pose_preds, gt_poses):
        return nn.MSELoss()(pose_preds, gt_poses)