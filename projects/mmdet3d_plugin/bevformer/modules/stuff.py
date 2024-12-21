# Reprojection Error

def multi_camera_reprojection_error_single_flow(K_list, T_cam_to_ego_list, T_ego_to_global, pose_ego, depth_list, flow, img1_list, img2_list):
    """
    Compute the reprojection error between two time steps across multiple camera views with a single optical flow.
    
    Args:
        K_list (list): List of intrinsic matrices (6 cameras, each 3x3).
        T_cam_to_ego_list (list): List of extrinsic matrices (6 cameras, each 4x4).
        T_ego_to_global (torch.Tensor): Ego pose in global frame at the current time step (4x4).
        pose_ego (torch.Tensor): Ego pose transformation between the two time steps (4x4).
        depth_list (list): List of depth maps for each camera at the first time step (6 depth maps).
        flow (torch.Tensor): Single optical flow between frames for all cameras (N, 2, H, W).
        img1_list (list): List of images from the first time step (6 images).
        img2_list (list): List of images from the second time step (6 images).
    
    Returns:
        torch.Tensor: Average reprojection error across all camera views.
    """
    total_error = 0.0
    num_cameras = len(K_list)  # Should be 6 for nuScenes
    N, C, H, W = img1_list[0].shape  # Assuming all images are the same size
    
    # Loop through all cameras
    for cam_idx in range(num_cameras):
        # Camera-specific intrinsics and extrinsics
        K = K_list[cam_idx]
        T_cam_to_ego = T_cam_to_ego_list[cam_idx]
        
        # Depth and images for this camera
        depth = depth_list[cam_idx]
        img1 = img1_list[cam_idx]
        img2 = img2_list[cam_idx]

        # Generate mesh grid of pixel coordinates
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid_x = grid_x.to(depth.device).float()
        grid_y = grid_y.to(depth.device).float()
        pixels = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # Shape: (H*W, 2)
        
        # Backproject pixels into 3D using depth and intrinsics
        pixels_homogeneous = torch.cat([pixels, torch.ones(H * W, 1).to(pixels.device)], dim=-1)  # (H*W, 3)
        points_3d = torch.matmul(torch.inverse(K), pixels_homogeneous.T)  # (3, H*W)
        points_3d = points_3d.T * depth.view(-1, 1)  # Multiply by depth to get 3D points
        
        # Transform points from camera to ego frame
        points_3d_homogeneous = torch.cat([points_3d, torch.ones(H * W, 1).to(points_3d.device)], dim=-1)  # (H*W, 4)
        points_ego = torch.matmul(T_cam_to_ego, points_3d_homogeneous.T).T  # (H*W, 4)

        # Transform points from ego frame to global frame
        points_global = torch.matmul(T_ego_to_global, points_ego.T).T  # (H*W, 4)
        
        # Apply the ego motion between time steps
        points_ego_next = torch.matmul(pose_ego, points_global.T).T  # Transform to next time step in ego frame
        
        # Project the points back to the image plane of the second camera
        points_3d_next = torch.matmul(torch.inverse(T_cam_to_ego), points_ego_next.T).T  # Back to camera frame
        projected_pixels = torch.matmul(K, points_3d_next[:, :3].T).T  # Project to 2D
        projected_pixels = projected_pixels[:, :2] / projected_pixels[:, 2:]  # Normalize by z

        # Apply the same optical flow for all cameras
        flow_flat = flow.permute(0, 2, 3, 1).reshape(-1, 2)  # (H*W, 2)
        projected_pixels_with_flow = pixels + flow_flat  # Apply flow

        # Reshape for error calculation
        projected_pixels = projected_pixels.view(H, W, 2)
        projected_pixels_with_flow = projected_pixels_with_flow.view(H, W, 2)

        # Compute reprojection error for this camera
        error = F.mse_loss(projected_pixels, projected_pixels_with_flow)
        total_error += error

    # Average the reprojection error across all cameras
    avg_reprojection_error = total_error / num_cameras
    return avg_reprojection_error
