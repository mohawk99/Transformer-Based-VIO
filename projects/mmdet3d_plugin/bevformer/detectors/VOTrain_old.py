import torch
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
#nusc = NuScenes(version='v1.0-mini', dataroot='/content/drive/My Drive/Thesis/PanoOcc/data/occ3d-nus/', verbose=True)
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/mohak/Thesis/PanoOcc/data/occ3d-nus/', verbose=True)




@DETECTORS.register_module()
class VOTrain(MVXTwoStageDetector):
    """PanoOcc.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
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
        # loss_inputs = [gt_bboxes_3d, gt_labels_3d, voxel_semantics, mask_camera, outs]
        # losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return outs['bev_embed']

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

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
        for i in range(len(pred_trans_list)):
            # Current pose
            curr_pos = gt_poses[i+1, :3]
            curr_quat = gt_poses[i+1, 3:]
            
            # Calculate relative position in reference frame
            delta_pos = torch.matmul(ref_R.transpose(0,1), (curr_pos - ref_pos))
            
            # Calculate relative orientation
            delta_quat = quaternion_multiply(
                quaternion_conjugate(ref_quat), 
                curr_quat
            )
            
            # Losses
            trans_loss = F.l1_loss(pred_trans_list[i], delta_pos)
            #rot_loss = quaternion_distance_loss(pred_rot_list[i], delta_quat)
            
            total_loss += trans_loss# + rot_loss
            
            # Update reference for next iteration
            ref_pos = curr_pos
            ref_quat = curr_quat
            ref_R = quaternion_to_rotation_matrix(ref_quat)

        return total_loss / len(pred_trans_list)

    def quaternion_distance_loss(pred, target):
        # Geodesic distance between quaternions
        dot_product = torch.sum(pred * target, dim=1).clamp(-1, 1)
        return 2 * torch.acos(torch.abs(dot_product)).mean()

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

        len_queue = img.size(1)
        losses = dict()

        all_pred_trans = []
        all_pred_rots = []

        for i in range(len_queue - 1):
            curr_pair = img[:, i:i+2, ...]
            pair_img_metas = [each[i:i+2] for each in img_metas]
            
            # Process first image of pair
            prev_frames = img[:, :i, ...]
            if prev_frames.size(1) > 0:
                prev_img_metas = copy.deepcopy([each[:i] for each in img_metas])
                prev_bev_first = self.obtain_history_bev(prev_frames, prev_img_metas)
            else:
                prev_bev_first = None
                
            first_img = curr_pair[:, 0, ...]
            first_img_meta = [each[0] for each in pair_img_metas]
            
            if not first_img_meta[0]['prev_bev_exists']:
                prev_bev_first = None
                
            first_img_feats = self.extract_feat(img=first_img, img_metas=first_img_meta)

            
            # Process second image of pair
            prev_frames_second = img[:, :i+1, ...]
            prev_img_metas_second = copy.deepcopy([each[:i+1] for each in img_metas])
            prev_bev_second = self.obtain_history_bev(prev_frames_second, prev_img_metas_second)
            
            second_img = curr_pair[:, 1, ...]
            second_img_meta = [each[1] for each in pair_img_metas]
            
            if not second_img_meta[0]['prev_bev_exists']:
                prev_bev_second = None
                
            second_img_feats = self.extract_feat(img=second_img, img_metas=second_img_meta)

            
            # Get occupancy for previous frame
            # prev_data = self.pts_bbox_head(
            #     first_img_feats, first_img_meta, 
            #     prev_bev=prev_bev_first)
            
            prev_data = self.forward_pts_train(first_img_feats, gt_bboxes_3d,
                                            gt_labels_3d, voxel_semantics, mask_camera, first_img_meta,
                                            gt_bboxes_ignore, prev_bev_first)
            
            # Get occupancy for current frame
            # curr_data = self.pts_bbox_head(
            #     second_img_feats, second_img_meta, 
            #     prev_bev=prev_bev_second)
            
            curr_data = self.forward_pts_train(second_img_feats, gt_bboxes_3d,
                                            gt_labels_3d, voxel_semantics, mask_camera, second_img_meta,
                                            gt_bboxes_ignore, prev_bev_second)
            
            # Estimate relative pose between frames
            trans, rots = self.pose_estimator(prev_data, curr_data)
            all_pred_trans.append(trans)
            all_pred_rots.append(rots)
            

        gt_poses = getvoposes(nusc, img_metas, '/content/drive/My Drive/Thesis/PanoOcc/checkpoints/norm_gt_poses.json')
        device = trans.device
        gt_poses = gt_poses.to(device)
        gt_translation, gt_quaternion = gt_poses[:, :3], gt_poses[:, 3:]

        losses['translation_loss'] = self.compute_loss(all_pred_trans, all_pred_rots, gt_translation, gt_quaternion)

        return losses

    def forward_test(self, img_metas,
                     img=None,
                     voxel_semantics=None,
                     mask_lidar=None,
                     mask_camera=None,
                     **kwargs):
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
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, occ = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        # for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
        #     result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, occ
