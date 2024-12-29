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
nusc = NuScenes(version='v1.0-mini', dataroot='/content/drive/MyDrive/Thesis/PanoOcc/data/occ3d-nus/', verbose=True)


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
        
        self.transformer_dim = 256

        self.pose_head = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim, self.transformer_dim // 2),
            nn.GELU(),
            nn.Linear(self.transformer_dim // 2, self.transformer_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer_dim // 4, 7)  # Output: 3 for translation, 4 for rotation (quaternion)
        )

        self.optical_flow_head = OpticalFlowHead(in_channels= 512)

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

        num_sequences = img.size(1)

        all_pred_poses = []

        # Iterate through img_metas sequences
        for sequence_idx in range(num_sequences):  # img_metas contains 4 sequences
            # Extract img_feats for the current sequence
            #print("Pose loss current meta:", sequence_img_meta)
            current_img = img[:, sequence_idx, :, :, :, :]
            img_feats = self.extract_feat(current_img)

            # Initialize list to collect pose predictions for the sequence
            pred_poses = []

            # Iterate through time steps (frames) in the sequence
            B, T, C, H, W = img_feats[-1].shape  # Example shape: [Batch, Time, Channels, Height, Width]

            # Flatten spatial dimensions (H, W)
            img_feats_flat = img_feats[-1].view(B, T, C, -1).mean(dim=-1)  # Shape: [B, T, C]

            # Pool temporal features if needed
            img_feats_pooled = img_feats_flat.mean(dim=1)  # Shape: [B, C]

            # Predict poses using pose_head
            pose_output = self.pose_head(img_feats_pooled)

            pred_poses.append(pose_output)

            # Stack predicted poses for the sequence
            all_pred_poses = torch.stack(pred_poses, dim=1)  # Shape: [B, T, 7]
        # Concatenate predicted poses from all sequences
        pose_preds = all_pred_poses.view(-1, 7)

        # Ground truth poses
        gt_poses = getvoposes(nusc, img_metas)  # Shape: [B * T, 7]

        # Ensure devices match
        device = pose_preds.device
        gt_poses = gt_poses.to(device)


        # Pose loss
        pose_loss = F.mse_loss(pose_preds, gt_poses)
        losses['pose_loss'] = pose_loss