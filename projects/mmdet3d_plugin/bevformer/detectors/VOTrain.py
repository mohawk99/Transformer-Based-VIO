import torch
import gc
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()
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
from ..modules.PoseDecoder import *
from ..modules.getgtposes import getvoposes
from nuscenes.nuscenes import NuScenes


#Insert the dataset path 
nusc = NuScenes(version='v1.0-trainval', dataroot='', verbose=True)

#nusc = NuScenes(version='v1.0-mini', dataroot='/content/drive/My Drive/Thesis/PanoOcc/data/occ3d-nus/', verbose=True)
#nusc = NuScenes(version='v1.0-trainval', dataroot='/home/mohak/Thesis/PanoOcc/data/occ3d-nus/', verbose=True)


import math
import torch.nn.functional as F
@DETECTORS.register_module()
class VOTrain(MVXTwoStageDetector):
    """PanoOcc.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 PoseDecoder = None,
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
                 time_interval=1,
                 ):

        super(VOTrain,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.time_interval = time_interval

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': [],
            "ego2global_transform_lst": [],
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        self.pose_estimator = BEVPoseEstimator()

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            elif img.dim() == 6:
                B, T, V, C, H, W = img.size()
                img = img.view(B * T * V, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'),out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          img_metas,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        #bev_data = outs['bev_embed']

        return outs


    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        is_training = self.training
        self.eval()

        prev_bev_lst = []
        with torch.no_grad():
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, only_bev=True)
                prev_bev = prev_bev.permute(0, 2, 1)
                prev_bev = prev_bev.reshape(prev_bev.shape[0], -1, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, self.pts_bbox_head.bev_z)
                prev_bev_lst.append(prev_bev)
        if is_training:
            self.train()
        # (bs, num_queue, embed_dims, H, W)
        return torch.stack(prev_bev_lst, dim=1)
    
    def compute_loss(self, pred_trans_list, pred_rot_list, gt_poses):
        device = pred_trans_list[0].device
        gt_poses = gt_poses.to(device)
        
        # Initialize first pose as reference
        ref_pos = gt_poses[0, :3]
        ref_quat = gt_poses[0, 3:]
        ref_R = quaternion_to_rotation_matrix(ref_quat)
        
        total_loss = 0
        t_loss = 0
        r_loss = 0
        for i in range(len(pred_trans_list)):
            # Current pose
            curr_pos = gt_poses[i+1, :3]
            curr_quat = gt_poses[i+1, 3:]
            
            # Calculate relative position in reference frame
            delta_pos = torch.matmul(ref_R.transpose(0,1), (curr_pos - ref_pos))
            delta_pos = delta_pos.unsqueeze(0)
            
            # Calculate relative orientation
            delta_quat = quaternion_multiply(
                quaternion_conjugate(ref_quat), 
                curr_quat
            )
            delta_quat = delta_quat.unsqueeze(0) 
            
            # Losses
            trans_loss = F.l1_loss(pred_trans_list[i], delta_pos)
            rot_loss = self.quaternion_distance_loss(pred_rot_list[i], delta_quat)
            
            #total_loss += trans_loss + rot_loss

            t_loss += trans_loss
            r_loss += rot_loss


            # Update reference for next iteration
            ref_pos = curr_pos
            ref_quat = curr_quat
            ref_R = quaternion_to_rotation_matrix(ref_quat)

        return t_loss / len(pred_trans_list), r_loss / len(pred_trans_list)

    def quaternion_distance_loss(self, pred, target):
        # Geodesic distance between quaternions
        dot_product = torch.sum(pred * target, dim=1).clamp(-1, 1)
        return 2 * torch.acos(torch.abs(dot_product)).mean()
    
    def crop_and_pool_bev(self, bev_embed, target_size=(40,40)):
      # Ensure bev_embed is not a dictionary
      if isinstance(bev_embed, dict):
          bev_embed = bev_embed['bev_embed']
      
      # Ensure we have the right shape
      if bev_embed.dim() == 3:
          bs = bev_embed.size(0)
          hw = bev_embed.size(1)
          embed_dims = bev_embed.size(2)
      else:
          bs, hw, embed_dims = 1, bev_embed.size(0), bev_embed.size(1)
      
      h = w = int(math.sqrt(hw))
      
      # Reshape and center crop
      bev_embed = bev_embed.view(bs, h, w, embed_dims)
      h_start = (h - target_size[0]) // 2
      h_end = h_start + target_size[0]
      w_start = (w - target_size[1]) // 2
      w_end = w_start + target_size[1]
      
      cropped_bev = bev_embed[:, h_start:h_end, w_start:w_end, :]
      
      # Restore pooling
      cropped_bev = cropped_bev.permute(0, 3, 1, 2)  # (bs, embed_dims, h, w)
      pooled_bev = F.adaptive_avg_pool2d(cropped_bev, output_size=(target_size[0], target_size[1]))
      pooled_bev = pooled_bev.permute(0, 2, 3, 1)  # (bs, h, w, embed_dims)
      
      cropped_bev = pooled_bev.reshape(bs, -1, embed_dims)
      
      return cropped_bev
    
    
    @auto_fp16(apply_to=('img', 'prev_bev'))
    def forward(self,
                points=None,
                img_metas=None,
                gt_bboxes_3d=None,
                gt_labels_3d=None,
                voxel_semantics=None,
                mask_lidar=None,
                mask_camera=None,
                gt_labels=None,
                gt_bboxes=None,
                img=None,
                proposals=None,
                gt_bboxes_ignore=None,
                img_depth=None,
                img_mask=None,
                ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        torch.cuda.empty_cache()
        losses = dict()
        len_queue = img.size(1)
        all_bev_data = []
        
        # First pass: Generate and store all BEV embeddings
        for i in range(len_queue):
            curr_img = img[:, i, ...]
            curr_img_meta = [each[i] for each in img_metas]
            
            # Extract features and get BEV data
            curr_img_feats = self.extract_feat(img=curr_img, img_metas=curr_img_meta)
            curr_data = self.pts_bbox_head(curr_img_feats, curr_img_meta, prev_bev=None)
            
            # Move to CPU and store
            cpu_data = {
                'bev_embed': curr_data['bev_embed'].cpu(),
                'frame_idx': i
            }
            all_bev_data.append(cpu_data)
            
            # Clear GPU memory
            del curr_img_feats, curr_data
            torch.cuda.empty_cache()

        # Second pass: Process pairs for pose estimation
        all_pred_trans = []
        all_pred_rots = []
        
        for i in range(len_queue - 1):
            # Load consecutive frames back to GPU
            prev_data = {'bev_embed': all_bev_data[i]['bev_embed'].cuda()}
            curr_data = {'bev_embed': all_bev_data[i+1]['bev_embed'].cuda()}
            
            # Estimate pose
            trans, rots = self.pose_estimator(prev_data['bev_embed'], curr_data['bev_embed'])
            
            all_pred_trans.append(trans)
            all_pred_rots.append(rots)
            
            # Clear GPU memory
            del prev_data, curr_data
            torch.cuda.empty_cache()

        # Compute loss
        gt_poses = getvoposes(nusc, img_metas, '/checkpoints/norm_gt_poses.json')
        losses['translation_loss'],losses['quat_loss'] = self.compute_loss(all_pred_trans, all_pred_rots, gt_poses)

        return losses