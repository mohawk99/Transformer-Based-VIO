import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import gc
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from ..modules.PoseDecoder import BEVPoseEstimator, quaternion_to_rotation_matrix, quaternion_multiply, quaternion_conjugate
from ..modules.getgtposes import getvoposes
import math
from nuscenes.nuscenes import NuScenes

@DETECTORS.register_module()
class VOTrain(MVXTwoStageDetector):
    """PanoOcc with Pose Estimation using curriculum learning.
    
    Combines the occupancy prediction capabilities of PanoOcc with
    pose estimation from VOTrain using curriculum learning to
    gradually shift focus between the two tasks.
    """

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
                 time_interval=1,
                 # Curriculum learning parameters
                 curriculum_learning=True,
                 curriculum_epochs=24,  # Total expected training epochs
                 init_pose_weight=0.1,  # Starting weight for pose loss
                 max_pose_weight=1.0,   # Maximum weight for pose loss
                 nusc_dataroot=None,    # Path to nuScenes data for pose GT
                 ):

        super(VOTrain, self).__init__(
            pts_voxel_layer, pts_voxel_encoder,
            pts_middle_encoder, pts_fusion_layer,
            img_backbone, pts_backbone, img_neck, pts_neck,
            pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained)
            
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.time_interval = time_interval

        # Temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': [],
            "ego2global_transform_lst": [],
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        # Pose estimation related
        self.pose_estimator = BEVPoseEstimator()
        self.nusc_dataroot = nusc_dataroot
        
        # Try loading NuScenes if path is provided
        self.nusc = NuScenes(version='v1.0-mini', dataroot='/content/drive/My Drive/Thesis/PanoOcc/data/occ3d-nus/', verbose=True)
        # if nusc_dataroot:
        #     try:
        #         from nuscenes.nuscenes import NuScenes
        #         self.nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_dataroot, verbose=True)
        #     except Exception as e:
        #         print(f"Warning: Could not load NuScenes data: {e}")
        #         print("Will need to provide NuScenes instance externally or through gt_poses argument")
        
        # Curriculum learning setup
        self.curriculum_learning = curriculum_learning
        self.curriculum_epochs = curriculum_epochs
        self.init_pose_weight = init_pose_weight
        self.max_pose_weight = max_pose_weight
        self.current_epoch = 0
        
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
                          gt_bboxes_3d,
                          gt_labels_3d,
                          voxel_semantics,
                          mask_camera,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function for PanoOcc part."""
        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, voxel_semantics, mask_camera, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses, outs

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively."""
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
                prev_bev = prev_bev.reshape(prev_bev.shape[0], -1, 
                                           self.pts_bbox_head.bev_h, 
                                           self.pts_bbox_head.bev_w, 
                                           self.pts_bbox_head.bev_z)
                prev_bev_lst.append(prev_bev)
        if is_training:
            self.train()
        # (bs, num_queue, embed_dims, H, W)
        return torch.stack(prev_bev_lst, dim=1)
        
    def compute_pose_loss(self, pred_trans_list, pred_rot_list, gt_poses):
        """Compute pose estimation loss."""
        device = pred_trans_list[0].device
        gt_poses = gt_poses.to(device)
        
        # Initialize first pose as reference
        ref_pos = gt_poses[0, :3]
        ref_quat = gt_poses[0, 3:]
        ref_R = quaternion_to_rotation_matrix(ref_quat)
        
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
            
            t_loss += trans_loss
            r_loss += rot_loss

            # Update reference for next iteration
            ref_pos = curr_pos
            ref_quat = curr_quat
            ref_R = quaternion_to_rotation_matrix(ref_quat)

        return t_loss / len(pred_trans_list), r_loss / len(pred_trans_list)

    def quaternion_distance_loss(self, pred, target):
        """Geodesic distance between quaternions."""
        dot_product = torch.sum(pred * target, dim=1).clamp(-1, 1)
        return 2 * torch.acos(torch.abs(dot_product)).mean()
    
    def crop_and_pool_bev(self, bev_embed, target_size=(40,40)):
        """Process BEV embeddings for pose estimation."""
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

    def get_curriculum_weight(self):
        """Calculate curriculum weight based on current epoch."""
        if not self.curriculum_learning:
            return self.max_pose_weight
        
        # Linear increase from init_weight to max_weight
        progress = min(1.0, self.current_epoch / self.curriculum_epochs)
        weight = self.init_pose_weight + (self.max_pose_weight - self.init_pose_weight) * progress
        return weight
        
    def set_epoch(self, epoch):
        """Set current epoch for curriculum learning."""
        self.current_epoch = epoch
    
    def forward_train(self,
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
                      gt_poses=None):
        """Forward training function with curriculum learning."""
        gc.collect()
        torch.cuda.empty_cache()
        

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img_pano = img[:, -1, ...]

        if prev_img.size(1)==0:
            prev_bev = None
        else:
            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas_pano = [each[len_queue - 1] for each in img_metas]
        if not img_metas_pano[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img_pano, img_metas=img_metas_pano)
        
        # Part 1: Original PanoOcc losses
        pano_losses = self.forward_pts_train(
            img_feats, gt_bboxes_3d, gt_labels_3d, 
            voxel_semantics, mask_camera, img_metas_pano,
            gt_bboxes_ignore, prev_bev)
        
        losses = dict()
        losses.update(pano_losses)

        del img_feats, prev_img, img_metas_pano, img_pano
        gc.collect()
        torch.cuda.empty_cache()
        
        # Part 2: Pose estimation if we have multiple frames
        if len_queue > 1:
            # Free memory
            gc.collect()
            torch.cuda.empty_cache()
            
            
            # Process sequence for pose estimation
            all_bev_data = []
            all_pred_trans = []
            all_pred_rots = []
            
            # First pass: Generate and store all BEV embeddings
            for i in range(len_queue):
                curr_img = img[:, i, ...]
                curr_img_meta = [each[i] for each in img_metas]
                
                # Extract features and get BEV data
                curr_img_feats = self.extract_feat(img=curr_img, img_metas=curr_img_meta)
                out_data = self.pts_bbox_head(curr_img_feats, curr_img_meta, prev_bev=None)
                if i == len_queue-1:
                    # Reuse outputs we already computed
                    curr_data = {'bev_embed': out_data['bev_pose']}
                else:
                    curr_data = self.pts_bbox_head(curr_img_feats, curr_img_meta, prev_bev=None)
                
                # Move to CPU to save memory
                cpu_data = {
                    'bev_embed': curr_data['bev_embed'].cpu(),
                    'frame_idx': i
                }
                all_bev_data.append(cpu_data)
                
                # Clear GPU memory if not the last frame data we need
                if i != len_queue-1:
                    del curr_img_feats, curr_data
                    gc.collect()
                    torch.cuda.empty_cache()

            # Second pass: Process pairs for pose estimation
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
                gc.collect()
                torch.cuda.empty_cache()

            # Get ground truth poses
            if gt_poses is None:
                if self.nusc is not None:
                    gt_poses_path = '/checkpoints/norm_gt_poses.json'  # Change to your actual path
                    gt_poses = getvoposes(self.nusc, prev_img_metas, gt_poses_path)
                else:
                    raise ValueError("No gt_poses provided and NuScenes instance not available")
            
            # Compute pose loss with curriculum weighting
            t_loss, r_loss = self.compute_pose_loss(all_pred_trans, all_pred_rots, gt_poses)
            
            # Apply curriculum weighting
            #weight = self.get_curriculum_weight()
            weight = 1.0
            losses['translation_loss'] = t_loss * weight
            losses['quaternion_loss'] = r_loss * weight

        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        # Implementation similar to PanoOcc's forward_test
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = []
            self.prev_frame_info["ego2global_transformation_lst"] = []
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = []
            self.prev_frame_info["ego2global_transformation_lst"] = []

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        self.prev_frame_info["ego2global_transformation_lst"].append(img_metas[0][0]["ego2global_transformation"])

        img_metas[0][0]["ego2global_transform_lst"] = self.prev_frame_info["ego2global_transformation_lst"][-1::-self.time_interval][::-1]
        prev_bev = self.prev_frame_info['prev_bev'][-self.time_interval:: -self.time_interval][:: -1]
        prev_bev = torch.stack(prev_bev, dim=1) if len(prev_bev) > 0 else None

        new_prev_bev, occ_results = self.simple_test(
            img_metas[0], img[0], prev_bev=prev_bev, **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.

        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        new_prev_bev = new_prev_bev.permute(0, 2, 1).reshape(1, -1, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, self.pts_bbox_head.bev_z)
        self.prev_frame_info['prev_bev'].append(new_prev_bev)

        while len(self.prev_frame_info["prev_bev"]) >= self.pts_bbox_head.transformer.temporal_encoder.num_bev_queue * self.time_interval:
            self.prev_frame_info["prev_bev"].pop(0)
            self.prev_frame_info["ego2global_transformation_lst"].pop(0)

        return occ_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev, test=True)

        occ = self.pts_bbox_head.get_occ(
            outs, img_metas, rescale=rescale)

        return outs['bev_embed'], occ

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentation."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        new_prev_bev, occ = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        return new_prev_bev, occ
        
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)